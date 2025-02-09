from model import ModelConfig, LlamaModel
from train import TrainerConfig, FileDataLoader, Trainer

from torch.distributed import init_process_group, destroy_process_group
from transformers import AutoTokenizer

import os


def main():
    tokenizer_id = "HuggingFaceTB/SmolLM-360M"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    tokenizer.pad_token = tokenizer.eos_token

    model_config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=960,
        d_head=64,
        d_mlp_proj=2560,
        n_layers=32,
        n_kv_heads=5,
        n_attn_heads=15,
        rms_norm_eps=1e-5,
        initializer_range=0.008,
        rope_theta=100000.0,
        padding_idx=tokenizer.pad_token_id
    )

    train_config = TrainerConfig(
        per_device_train_batch_size=32,
        grad_accumulation_steps=8,
        max_seq_len=2048,
        num_epochs=3,
        learning_rate=1e-3,
        grad_clip_norm=1.0,
        tokens_folder="math_tok",
        log_dir="runs/math_3epoch",
        warmup_ratio=0.02,
        val_size=0.001,
        eval_interval_steps=500,
        ddp=True,
        use_compile=True,
        checkpoint_save_interval=5000,
        checkpoint_dir_path="math_chkpts"
    )

    model = LlamaModel(model_config)
    dataloader = FileDataLoader(train_config, tokenizer, int(os.environ['WORLD_SIZE']), int(os.environ['LOCAL_RANK']))

    init_process_group("nccl")
    try:
        trainer = Trainer(train_config, model, tokenizer)
    
        trainer.train(dataloader)
    
        if trainer.main_process:
            trainer.save_checkpoint("math_3epoch")
    finally:
        destroy_process_group()


# Run:
# torchrun --standalone --nproc_per_node=4 run_ddp_train-math.py > output.log 2>&1 &
# tail -f output.log
if __name__ == "__main__":
    main()