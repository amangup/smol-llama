from think_model import ThinkModelConfig, ThinkTransformer
from train import TrainerConfig, FileDataLoader, Trainer

from torch.distributed import init_process_group, destroy_process_group
from transformers import AutoTokenizer

import os


def main():
    tokenizer_id = "HuggingFaceTB/SmolLM-360M"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    tokenizer.pad_token = tokenizer.eos_token

    model_config = ThinkModelConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=960,
        d_head=64,
        d_mlp_proj=2560,
        n_generate_layers=12,
        n_think_layers=32,
        n_kv_heads=5,
        n_attn_heads=15,
        rms_norm_eps=1e-5,
        initializer_range=0.002,
        rope_theta=100000.0,
        padding_idx=tokenizer.pad_token_id
    )

    train_config = TrainerConfig(
        per_device_train_batch_size=16,
        grad_accumulation_steps=32,
        max_seq_len=1024,
        num_epochs=1,
        learning_rate=4e-3,
        grad_clip_norm=1.0,
        tokens_folder="fineweb-edu_tok-100BT",
        log_dir="runs/think_fineweb-edu_100BT_3",
        warmup_ratio=0.02,
        val_size=0.0005,
        eval_interval_steps=500,
        ddp=True,
        use_compile=True,
        checkpoint_save_interval=5000,
        checkpoint_dir_path="think_fineweb-edu_chkpts_exp3"
    )

    model = ThinkTransformer(model_config)
    dataloader = FileDataLoader(train_config, tokenizer, int(os.environ['WORLD_SIZE']), int(os.environ['LOCAL_RANK']))

    init_process_group("nccl")
    try:
        trainer = Trainer(train_config, model, tokenizer)
    
        trainer.train(dataloader)
    
        if trainer.main_process:
            trainer.save_checkpoint("think_fineweb-edu")
    finally:
        destroy_process_group()


# Run:
# torchrun --standalone --nproc_per_node=4 run_ddp_train-think-fineweb-edu.py > think_output.log 2>&1 &
# tail -f think_output.log
if __name__ == "__main__":
    main()