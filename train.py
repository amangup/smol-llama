import time

from model import ModelConfig, LlamaModel

from dataclasses import dataclass
from datatrove.utils.dataset import DatatroveFolderDataset
from transformers import AutoTokenizer

import math
import os
import torch


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


class DataLoader:
    def __init__(self, config, tokenizer, text):
        self.config = config
        self.tokenizer = tokenizer
        self.batch_tensor_shape = (config.per_device_train_batch_size, config.max_seq_len)
        self.tokens = self._tokenize(text)
        print(f"{'Total train tokens':<30} | {self.tokens.numel() - self.tokens.size(0):,}")

        self.num_seqs = self.tokens.size(0)
        self.index = 0

    def next_batch(self):
        new_index = self.index + self.config.per_device_train_batch_size
        x = self.tokens[self.index:new_index, :-1].contiguous()
        y = self.tokens[self.index:new_index, 1:].contiguous()

        self.index = new_index
        if new_index > self.num_seqs:
            self.index = 0

        return x, y

    def num_steps_per_epoch(self):
        return math.ceil(self.num_seqs / self.config.per_device_train_batch_size)

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


class FileDataLoader:
    def __init__(self, config, tokenizer):
        self.config = config
        self.dataset = DatatroveFolderDataset(
            folder_path=config.tokens_folder,
            filename_pattern=os.path.join(config.tokens_folder, "*.ds"),
            seq_len=config.max_seq_len,
            token_size=(2 if tokenizer.vocab_size < 65535 else 4),
            recursive=False
        )
        self.num_seqs = len(self.dataset)

        print(f"{'Total train tokens':<30} | {self.num_seqs * config.max_seq_len:,}")

        self.index = 0

    def next_batch(self):
        new_index = self.index + self.config.per_device_train_batch_size

        x, y = zip(*[(self.dataset[idx]['input_ids'][:-1], self.dataset[idx]['input_ids'][1:])
                     for idx in range(self.index, min(new_index, self.num_seqs))])
        x_t = torch.stack(list(x))
        y_t = torch.stack(list(y))

        self.index = new_index
        if new_index > self.num_seqs:
            self.index = 0

        return x_t, y_t

    def num_steps_per_epoch(self):
        return math.ceil(self.num_seqs / self.config.per_device_train_batch_size)



class Trainer:
    def __init__(self, config, model):
        self.config = config
        self.model = model
        device_name, device_model, n_gpus = "cpu", "_", 1
        if torch.cuda.is_available():
            device_name = "cuda"
            device_model = torch.cuda.get_device_name()
            n_gpus = torch.cuda.device_count()

        self.device = torch.device(device_name)
        if device_name != "cpu":
            self.model.to(self.device)

        use_compile = self.config.use_compile and device_name == "cuda" and torch.__version__.startswith("2")
        if use_compile:
            self.model = torch.compile(self.model)

        self.dtype = getattr(torch, self.config.amp_dtype)

        print(f"{'Num Trainable Params':<30} | {self._num_trainable_params():,}")
        print(f"{'Train device':<30} | {self.device}, {device_model}, N={n_gpus}")
        print(f"{'Training precision':<30} | {self.dtype}")
        print(f"{'Flash Attention':<30} | {self.model.using_flash_attention()}")
        print(f"{'torch.compile()':<30} | {use_compile}")
        print("\n\n")


    def train(self, dataloader):
        steps_per_epoch = dataloader.num_steps_per_epoch()
        num_steps = steps_per_epoch * self.config.num_epochs
        if self.config.max_steps:
            num_steps = min(num_steps, self.config.max_steps)
        print(f"{'Training steps':<30} | {num_steps:,} ")

        optimizer = torch.optim.AdamW(self.model.parameters(),
                                      lr=self.config.learning_rate,
                                      fused=(self.device.type == "cuda"))

        for step in range(num_steps):
            X, Y = dataloader.next_batch()
            X, Y = X.to(self.device), Y.to(self.device)
            num_tokens = torch.numel(X)
            start = time.perf_counter()

            with torch.autocast(device_type=self.device.type, dtype=self.dtype):
                _, loss = self.model(X, Y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            end = time.perf_counter()
            tokens_per_sec = num_tokens / (end - start)
            print(f"Step: {step}, Training Loss: {loss.item():.5f}, Tokens/sec: {tokens_per_sec}")


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
        x, y = dataloader.next_batch()
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
