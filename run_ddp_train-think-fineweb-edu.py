from think_model import ThinkModelConfig, ThinkTransformer
from train import TrainerConfig, FileDataLoader, Trainer

from huggingface_hub import snapshot_download
from safetensors import safe_open
from torch.distributed import init_process_group, destroy_process_group
from transformers import AutoTokenizer

import os

def load_smollm_to_think_network(hf_id, model):
    local_path = snapshot_download(
        repo_id=hf_id,
        allow_patterns=[
            "config.json",
            "*.safetensors"
        ],
    )
    hf_state_dict = {}
    with safe_open(f'{local_path}/model.safetensors', framework='pt') as f:
        for key in f.keys():
            hf_state_dict[key.replace("model", "think_network")] = f.get_tensor(key)

    missing, unexpected = model.load_state_dict(hf_state_dict, strict=False)
    #print("missing keys: ", missing)
    #print("unexpected keys: ", unexpected)

def main():
    hf_id = "HuggingFaceTB/SmolLM2-360M"
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    tokenizer.pad_token = tokenizer.eos_token

    model_config = ThinkModelConfig(
        vocab_size=tokenizer.vocab_size,
        #
        # Generate model
        d_model=960,
        d_head=64,
        d_mlp_proj=2560,
        n_generate_layers=12,
        n_kv_heads=5,
        n_attn_heads=15,
        n_cross_attn_heads=15,
        generate_initializer_range=0.002,
        #
        # Think model
        think_d_model=960,
        think_d_head=64,
        think_d_mlp_proj=2560,
        n_think_kv_heads=5,
        n_think_attn_heads=15,
        n_think_layers=32,
        think_initializer_range=0.02,
        #
        # Others
        encode_interval=8,
        rms_norm_eps=1e-5,
        rope_theta=100000.0,
        padding_idx=tokenizer.pad_token_id
    )

    train_config = TrainerConfig(
        per_device_train_batch_size=32,
        grad_accumulation_steps=12,
        max_seq_len=1280,
        num_epochs=1,
        learning_rate=1e-3,
        grad_clip_norm=1.0,
        tokens_folder="fineweb-edu_tok-100BT",
        log_dir="runs/think_fineweb-edu_100BT_11",
        warmup_ratio=0.02,
        val_size=0.0005,
        eval_interval_steps=500,
        ddp=True,
        use_compile=True,
        checkpoint_save_interval=5000,
        checkpoint_dir_path="think_fineweb-edu_chkpts_exp11",
        plot_grad_norm=["think_network", "generate_network"],
        module_lrs={"think_network": 1e-2}
    )

    model = ThinkTransformer(model_config)
    load_smollm_to_think_network(hf_id, model)
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