from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.readers import HuggingFaceDatasetReader
from datatrove.pipeline.tokens.tokenizer import DocumentTokenizer
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoTokenizer

from model import ModelConfig, LlamaModel

import json
import torch
import secrets


def load_from_pretrained(model_id):
    local_dir = f"./hf_models/{model_id.split('/')[1]}"
    snapshot_download(
        repo_id=model_id,
        allow_patterns=[
            "config.json",
            "*.safetensors"
        ],
        local_dir=local_dir
    )

    with open(f'{local_dir}/config.json', 'r') as f:
        hf_config = json.load(f)

    model_config = ModelConfig(
        vocab_size=hf_config['vocab_size'],
        d_model=hf_config['hidden_size'],
        d_head=hf_config.get('head_dim', 64),
        d_mlp_proj=hf_config['intermediate_size'],
        n_layers=hf_config['num_hidden_layers'],
        n_kv_heads=hf_config['num_key_value_heads'],
        n_attn_heads=hf_config['num_attention_heads'],
        rms_norm_eps=hf_config['rms_norm_eps'],
        rope_theta=hf_config['rope_theta']
    )

    model = LlamaModel(model_config)
    hf_state_dict = {}
    with safe_open(f'{local_dir}/model.safetensors', framework='pt') as f:
        for k in f.keys():
            hf_state_dict[k[len("model."):]] = f.get_tensor(k)

    missing, unexpected = model.load_state_dict(hf_state_dict, strict=False)
    print(f"tensors missing: {missing}")
    print(f"tensors unexpected: {unexpected}")

    if 'lm_head.weight' in missing:
        with torch.no_grad():
            model.lm_head.weight = model.embed_tokens.weight

    return model


def save_to_hub(model, hf_repo):
    pass


def datatrove_tokenization_executor(hf_dataset_id, name, text_column, output_folder, tokenizer_id, eos_token, num_workers):
    pipeline = [
        HuggingFaceDatasetReader(
            dataset=hf_dataset_id,
            dataset_options={
                "split": 'train',
                "name": name,
            },
            text_key=text_column,
        ),
        DocumentTokenizer(
            output_folder=output_folder,
            tokenizer_name_or_path=tokenizer_id,
            eos_token=eos_token,
            batch_size=10000,
            max_tokens_per_file=int(1e8),
            shuffle=True,
            seed=1998
        )
    ]

    executor = LocalPipelineExecutor(
        pipeline=pipeline,
        logging_dir=f"logs_{secrets.token_hex(8)}/",
        tasks=num_workers,
    )

    return executor


def main():
    hf_checkpoint = "HuggingFaceTB/SmolLM2-135M"
    tokenizer = AutoTokenizer.from_pretrained(hf_checkpoint)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # input_ids = tokenizer(["Gravity is", "Dark Matter is"], return_tensors="pt").to(device)['input_ids']
    # model = load_from_pretrained(hf_checkpoint).to(device)
    # idx = model.generate(input_ids, temperature=0.25, top_k=25, max_new_tokens=16)
    # print(tokenizer.batch_decode(idx))

    executor = datatrove_tokenization_executor(
        hf_dataset_id="wikimedia/wikipedia",
        name="20231101.hi",
        text_column="text",
        output_folder="./wiki_hindi_tok",
        tokenizer_id=hf_checkpoint,
        eos_token=tokenizer.eos_token,
        num_workers=16
    )
    executor.run()


if __name__ == "__main__":
    main()

