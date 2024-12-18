import statistics
import time

from model import ModelConfig, LlamaModel

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datatrove.utils.dataset import DatatroveFolderDataset
from datetime import datetime
from transformers import AutoTokenizer
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

import math
import os
import torch
import torch.distributed as dist


@dataclass
class TrainerConfig:
    learning_rate: float = 1e-3
    per_device_train_batch_size: int = 32
    max_seq_len: int = 1024
    num_epochs: int = 1
    amp_dtype: str = 'bfloat16'
    use_compile: bool = True
    tokens_folder: str = 'tokens'
    max_steps: int = None
    eval_interval_steps: int = 16
    log_dir: str = 'runs'
    grad_clip_norm: float = None
    val_size: float = 0.1
    warmup_ratio: float = 0.01
    ddp: bool = False


class DataLoaderBase(ABC):
    def __init__(self, config: TrainerConfig):
        self.config = config

    def _num_steps(self, num_seqs):
        return math.ceil(num_seqs / self.config.per_device_train_batch_size)

    def _next_batch(self, index, start_pos, end_pos):
        new_index = index + self.config.per_device_train_batch_size

        x, y = self._get_x_y_tokens(index, new_index)

        index = new_index
        if new_index >= end_pos:
            index = start_pos

        return x, y, index

    @abstractmethod
    def _get_x_y_tokens(self, start, end):
        pass


class DataLoader(DataLoaderBase):
    def __init__(self, config, tokenizer, text):
        super().__init__(config)
        self.tokenizer = tokenizer
        self.batch_tensor_shape = (config.per_device_train_batch_size, config.max_seq_len)
        self.tokens = self._tokenize(text)
        print(f"{'Total tokens':<30} | {self.tokens.numel() - self.tokens.size(0):,}")

        self.num_seqs = self.tokens.size(0)
        self.train_seqs = math.ceil((1-config.val_size) * self.num_seqs)
        self.train_index = 0

        self.val_seqs = self.num_seqs - self.train_seqs
        self.val_index = self.train_seqs

    def next_batch_train(self):
        x, y, self.train_index = self._next_batch(self.train_index, 0, self.train_seqs)

        return x, y

    def next_batch_val(self):
        x, y, self.val_index = self._next_batch(self.val_index, self.train_seqs, self.num_seqs)

        return x, y

    def num_train_steps_per_epoch(self):
        return self._num_steps(self.train_seqs)

    def num_val_steps(self):
        return self._num_steps(self.val_seqs)

    def _tokenize(self, text):
        outputs = self.tokenizer(
            text,
            max_length=self.config.max_seq_len + 1,
            padding="max_length",
            truncation=True,
            return_overflowing_tokens=True,
            stride=1,
            return_tensors="pt",
        )

        tokens_tensor = outputs["input_ids"]
        shuffle_idx = torch.randperm(tokens_tensor.size(0))
        tokens_tensor = tokens_tensor[shuffle_idx, :]

        return tokens_tensor

    def _get_x_y_tokens(self, start, end):
        x = self.tokens[start:end, :-1].contiguous()
        y = self.tokens[start:end, 1:].contiguous()

        return x, y


class FileDataLoader(DataLoaderBase):
    def __init__(self, config, tokenizer, world_size=1, rank=0):
        self.config = config
        self.dataset = DatatroveFolderDataset(
            folder_path=config.tokens_folder,
            filename_pattern=os.path.join(config.tokens_folder, "*.ds"),
            seq_len=config.max_seq_len,
            token_size=(2 if tokenizer.vocab_size < 65535 else 4),
            recursive=False
        )

        self.num_seqs = len(self.dataset)
        if rank == 0:
            print(f"{'Total tokens':<30} | {self.num_seqs * config.max_seq_len:,}")

        self.total_train_seqs = math.ceil((1-config.val_size) * self.num_seqs)
        shard_size = self.total_train_seqs // world_size
        
        self.train_start_idx = rank * shard_size
        self.train_end_idx = (rank+1) * shard_size
        self.train_seqs = self.train_end_idx - self.train_start_idx
        
        print(f"Shard range rank:{rank:<13} | ({self.train_start_idx},{self.train_end_idx})")
        
        self.train_index = self.train_start_idx

        self.val_seqs = self.num_seqs - self.total_train_seqs
        self.val_index = self.total_train_seqs

    def next_batch_train(self):
        x, y, self.train_index = self._next_batch(self.train_index, self.train_start_idx, self.train_end_idx)

        return x, y

    def next_batch_val(self):
        x, y, self.val_index = self._next_batch(self.val_index, self.total_train_seqs, self.num_seqs)

        return x, y

    def num_train_steps_per_epoch(self):
        return self._num_steps(self.train_seqs)

    def num_val_steps(self):
        return self._num_steps(self.val_seqs)

    def _get_x_y_tokens(self, start, end):
        x, y = zip(*[(self.dataset[idx]['input_ids'][:-1], self.dataset[idx]['input_ids'][1:])
                     for idx in range(start, min(end, self.num_seqs))])
        x_t, y_t = torch.stack(list(x)), torch.stack(list(y))

        return x_t, y_t


class Trainer:
    def __init__(self, config, model):
        self.config = config
        self.model = model
        device_name, device_model, n_gpus = "cpu", "_", 1
        if torch.cuda.is_available():
            device_name = "cuda"
            device_model = torch.cuda.get_device_name()
            n_gpus = torch.cuda.device_count()

        use_compile = self.config.use_compile and device_name == "cuda" and torch.__version__.startswith("2")
        if use_compile:
            self.model = torch.compile(self.model)
        
        if config.ddp and device_name == "cuda" and n_gpus > 1:
            self.ddp = True
            self.local_rank = int(os.environ['LOCAL_RANK'])
            self.world_size = int(os.environ['WORLD_SIZE'])
            self.main_process = (self.local_rank == 0)
            
            self.device = torch.device(f"cuda:{self.local_rank}")
            torch.cuda.set_device(self.local_rank)
            
            self.model.to(self.device)
            self.model = DDP(self.model, device_ids=[self.local_rank])
        else:
            self.ddp = False
            self.main_process = True
            self.world_size = 1
            self.rank = 0
            if device_name != "cpu":
                self.device = torch.device(device_name)
                self.model.to(self.device)

        self.dtype = getattr(torch, self.config.amp_dtype)

        if self.main_process:
            print(f"{'Num Trainable Params':<30} | {self._num_trainable_params():,}")
            print(f"{'Train device':<30} | {self.device}, {device_model}, N={n_gpus}")
            print(f"{'Training precision':<30} | {self.dtype}")
            print(f"{'Flash Attention':<30} | {model.using_flash_attention()}")
            print(f"{'torch.compile()':<30} | {use_compile}")
            print(f"{'DistributedDataParallel':<30} | {self.ddp}")
            print("\n")


    def train(self, dataloader):
        steps_per_epoch = dataloader.num_train_steps_per_epoch()
        num_steps = steps_per_epoch * self.config.num_epochs
        if self.config.max_steps:
            num_steps = min(num_steps, self.config.max_steps)

        if self.main_process:
            print(f"{'Training steps':<30} | {num_steps:,} ")
            writer = SummaryWriter(log_dir=self.config.log_dir, flush_secs=30)

        optimizer = torch.optim.AdamW(self.model.parameters(),
                                      lr=self.config.learning_rate,
                                      betas=(0.9, 0.95),
                                      weight_decay=0.1,
                                      fused=(self.device.type == "cuda"))
        warmup_steps = math.floor(self.config.warmup_ratio * num_steps)
        warmup_factor = lambda st: (st+1) / warmup_steps
        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_factor)
        cos_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_steps-warmup_steps, eta_min=0.1*self.config.learning_rate
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer,
                                                          schedulers=[warmup_scheduler, cos_scheduler],
                                                          milestones=[warmup_steps])

        for step in range(num_steps):
            #print("begin step", self.local_rank)
            x, y = dataloader.next_batch_train()
            x, y = x.to(self.device), y.to(self.device)
            num_tokens = torch.numel(x)
            t_start = time.perf_counter()

            with torch.autocast(device_type=self.device.type, dtype=self.dtype):
                _, loss = self.model(x, y)

            loss.backward()
            if self.config.grad_clip_norm:
                norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)

            lr = scheduler.get_last_lr()[0]

            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            t_end = time.perf_counter()
            tokens_per_sec = (num_tokens*self.world_size) / (t_end-t_start)
            if self.main_process:
                # TODO: check if we can get true avg loss with minimal perf impact?
                print(f"Step: {step}, Training Loss: {loss.item():.5f}, LR: {lr:.7f}, Tokens/sec: {tokens_per_sec:.2f}")
                writer.add_scalar("train/loss", loss.item(), step)
                writer.add_scalar("train/learning_rate", lr, step)
                if self.config.grad_clip_norm:
                    writer.add_scalar("train/grad_norm", norm, step)

            # if run on step==0, training gets stuck
            if step == 3 or (step > 0 and step % self.config.eval_interval_steps == 0):
                if self.main_process:
                    eval_loss = self.eval(dataloader)
                    print(f"Step: {step}, Eval Loss: {eval_loss:.5f}")
                    writer.add_scalar("eval/loss", eval_loss, step)


    @torch.no_grad()
    def eval(self, dataloader):
        num_steps = dataloader.num_val_steps()
        print(f"Computing Eval loss, steps: {num_steps}")

        self.model.eval()

        loss_vals = []
        for step in range(num_steps):
            x, y = dataloader.next_batch_val()
            x, y = x.to(self.device), y.to(self.device)

            with torch.autocast(device_type=self.device.type, dtype=self.dtype):
                _, loss = self.model(x, y)
            loss_vals.append(loss.item())

        eval_loss = statistics.mean(loss_vals)
        self.model.train()

        return eval_loss

    def save_checkpoint(self, dir_path):
        os.makedirs(dir_path, exist_ok=True)
        timestamp = datetime.strftime(datetime.utcnow(), "%Y-%m-%d--%H-%M-%S")
        checkpoint_path = os.path.join(dir_path, f"model.checkpoint.{timestamp}")
        torch.save(self.model.state_dict(), checkpoint_path)
    
    def _num_trainable_params(self):
        return sum([p.data.numel() for p in self.model.parameters() if p.requires_grad])

    



def main():
    hf_checkpoint = "HuggingFaceTB/SmolLM2-135M"
    tokenizer = AutoTokenizer.from_pretrained(hf_checkpoint)
    tokenizer.pad_token = tokenizer.eos_token

    print(type(tokenizer))

    train_config = TrainerConfig(
        per_device_train_batch_size=8,
        max_seq_len=16,
        num_epochs=32
    )

    text = "Calm. Kindness. Kinship. Love. I've given up all chance at inner peace. I've made my mind a sunless space. I share my dreams with ghosts. I wake up every day to an equation I wrote 15 years ago from which there's only one conclusion, I'm damned for what I do. My anger, my ego, my unwillingness to yield, my eagerness to fight, they've set me on a path from which there is no escape. I yearned to be a savior against injustice without contemplating the cost and by the time I looked down there was no longer any ground beneath my feet. What is my sacrifice? I'm condemned to use the tools of my enemy to defeat them. I burn my decency for someone else's future. I burn my life to make a sunrise that I know I'll never see. And the ego that started this fight will never have a mirror or an audience or the light of gratitude. So what do I sacrifice? Everything! You'll stay with me, Lonni. I need all the heroes I can get."
    dataloader = DataLoader(train_config, tokenizer, text=text)
    for i in range(2):
        x, y = dataloader.next_batch_train()
        print(f"X: {x.shape} -", x)
        print(f"Y: {y.shape} -", y)

    model_config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=576,
        d_head=64,
        d_mlp_proj=1536,
        n_layers=30,
        n_kv_heads=3,
        n_attn_heads=9,
        rms_norm_eps=1e-5,
        initializer_range=0.041666666666666664,
        rope_theta=100000.0,
        padding_idx=tokenizer.pad_token_id
    )

    model = LlamaModel(model_config)
    trainer = Trainer(train_config, model)
    trainer.train(dataloader)

    input_ids = tokenizer(["I've given up all chance at"], return_tensors="pt")['input_ids'].cuda()
    idx = model.generate(input_ids, temperature=0.25, top_k=25, max_new_tokens=16)
    print(tokenizer.batch_decode(idx))


if __name__ == "__main__":
    main()
