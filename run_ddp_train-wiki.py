from model import ModelConfig, LlamaModel
from train import TrainerConfig, FileDataLoader, Trainer

from torch.distributed import init_process_group, destroy_process_group
from transformers import AutoTokenizer

import argparse
import os

def main():
    parser = argparse.ArgumentParser(description='Model training parameters')
    
    parser.add_argument('--n_layers', type=int, default=16,
                        help='Number of layers in the model')
    parser.add_argument('--d_model', type=int, default=576,
                        help='Model dimension/hidden size')
    parser.add_argument('--n_epochs', type=int, default=1,
                        help='Number of training epochs')
    
    args = parser.parse_args()
    
    tokenizer_id = "HuggingFaceTB/SmolLM2-135M"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    tokenizer.pad_token = tokenizer.eos_token

    n_layers=args.n_layers
    d_model=args.d_model
    n_epochs=args.n_epochs

    if d_model==576:
        d_mlp_proj=1536
        n_kv_heads=3
        n_attn_heads=9

    if d_model==960:
        d_mlp_proj=2560
        n_kv_heads=5
        n_attn_heads=15
        
    run_name = f"think_exp1_nothink_wiki_es_{n_layers}layer_{d_model}dim_lr1e-3_{n_epochs}epochs"
    model_config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        d_head=64,
        d_mlp_proj=d_mlp_proj,
        n_layers=n_layers,
        n_kv_heads=n_kv_heads,
        n_attn_heads=n_attn_heads,
        rms_norm_eps=1e-5,
        initializer_range=0.02,
        rope_theta=100000.0,
        padding_idx=tokenizer.pad_token_id
    )
    
    train_config = TrainerConfig(
        per_device_train_batch_size=64,
        max_seq_len=1024,
        num_epochs=n_epochs,
        eval_interval_steps=157,
        learning_rate=1e-3,
        grad_clip_norm=1.0,
        tokens_folder="wiki_es_tok",
        log_dir=f"runs/{run_name}",
        warmup_ratio=0.1,
        val_size=0.01,
        ddp=True,
        use_compile=True,
        test_generate_prefix="El mundo es"
    )

    model = LlamaModel(model_config)
    dataloader = FileDataLoader(train_config, tokenizer, int(os.environ['WORLD_SIZE']), int(os.environ['LOCAL_RANK']))

    init_process_group("nccl")
    try:
        trainer = Trainer(train_config, model, tokenizer)
    
        trainer.train(dataloader)

    finally:
        destroy_process_group()


# Run:
# torchrun --standalone --nproc_per_node=4 run_ddp_train-wiki.py > output.log 2>&1 &
# tail -f output.log
if __name__ == "__main__":
    main()