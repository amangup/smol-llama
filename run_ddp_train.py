from model import ModelConfig, LlamaModel
from train import TrainerConfig, FileDataLoader, Trainer

from torch.distributed import init_process_group, destroy_process_group
from transformers import AutoTokenizer

import os


def main():
    tokenizer_id = "HuggingFaceTB/SmolLM2-135M"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    tokenizer.pad_token = tokenizer.eos_token

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

    train_config = TrainerConfig(
        per_device_train_batch_size=32,
        max_seq_len=2048,
        num_epochs=1,
        eval_interval_steps=190*5,
        learning_rate=1e-3,
        grad_clip_norm=1.0,
        tokens_folder="fineweb-edu_tok-100BT",
        log_dir="runs/fineweb-100BT-exp2",
        warmup_ratio=0.01,
        val_size=0.0005,
        ddp=True,
        use_compile=True
    )

    model = LlamaModel(model_config)
    dataloader = FileDataLoader(train_config, tokenizer, int(os.environ['WORLD_SIZE']), int(os.environ['LOCAL_RANK']))

    init_process_group("nccl")
    try:
        trainer = Trainer(train_config, model, tokenizer)
    
        trainer.train(dataloader)
    
        if trainer.main_process:
            trainer.save_checkpoint("fineweb-100BT")
    finally:
        destroy_process_group()


# Run:
# torchrun --standalone --nproc_per_node=8 run_ddp_train.py > output.log 2>&1 &
# tail -f output.log
if __name__ == "__main__":
    main()