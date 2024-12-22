from bidict import bidict
from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.readers import HuggingFaceDatasetReader
from datatrove.pipeline.tokens.tokenizer import DocumentTokenizer
from huggingface_hub import snapshot_download, HfApi
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

from model import ModelConfig, LlamaModel

import json
import os
import secrets
import shutil
import torch


def _get_key_prefix(model_keys, weight_keys, tie_word_embeddings=False):
    model_keys = sorted(list(model_keys))
    weight_keys = sorted(list(weight_keys))
    if tie_word_embeddings:
        model_keys.remove("lm_head.weight")
        
    extra_prefix = [weight_key.replace(model_key, "") for model_key, weight_key in zip(model_keys, weight_keys)]
    is_same = all([prefix==extra_prefix[0] for prefix in extra_prefix])
    if not is_same:
        raise RuntimeError(f"weight keys are not in the format <prefix>.<model_key>")

    return extra_prefix[0]


# model_config -> HF config
CONFIG_KEYS_MAP = {
    "vocab_size": "vocab_size",
    "d_model": "hidden_size",
    "d_head": "head_dim",
    "d_mlp_proj": "intermediate_size",
    "n_kv_heads": "num_key_value_heads",
    "n_attn_heads": "num_attention_heads",
    "n_layers": "num_hidden_layers",
    "rms_norm_eps": "rms_norm_eps",
    "rope_theta": "rope_theta",
    "initializer_range": "initializer_range",
    "padding_idx": "pad_token_id",
    "tie_word_embeddings": "tie_word_embeddings"
}

def _model_config_from_hf(hf_config):
    kwargs = {model_key: hf_config[hf_key] for model_key, hf_key in CONFIG_KEYS_MAP.items() if hf_key in hf_config}
    return ModelConfig(**kwargs)


def _hf_config_from_model(model_config, tokenizer, dtype):
    hf_config = {hf_key: getattr(model_config, model_key) for model_key, hf_key in CONFIG_KEYS_MAP.items()}
    extras = {
        "architectures": [
            "LlamaForCausalLM"
        ],
        "attention_bias": False,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "hidden_act": "silu",
        "is_llama_config": True,
        "model_type": "llama",
        "max_position_embeddings": 8192,
        "torch_dtype": f"{dtype}".split(".")[1],
        "use_cache": True,
        "rope_scaling": None,
        "rope_interleaved": False
    }

    return hf_config | extras


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

    model_config = _model_config_from_hf(hf_config)
    model = LlamaModel(model_config)

    hf_state_dict = {}
    with safe_open(f'{local_dir}/model.safetensors', framework='pt') as f:
        prefix = _get_key_prefix(model.state_dict().keys(), f.keys(), hf_config['tie_word_embeddings'])
        for key in f.keys():
            hf_state_dict[key[len(prefix):]] = f.get_tensor(key)

    missing, unexpected = model.load_state_dict(hf_state_dict, strict=False)
    print(f"tensors missing: {missing}")
    print(f"tensors unexpected: {unexpected}")

    if hf_config['tie_word_embeddings'] and 'lm_head.weight' in missing:
        with torch.no_grad():
            model.lm_head.weight = model.embed_tokens.weight

    return model


def save_to_hub(model, tokenizer, hf_repo, checkpoint_path=None, private=False):
    # Model weights
    loaded_weights = model.state_dict()

    # If checkpoint_path, use those weights
    if checkpoint_path:
        chkpt_weights = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
        
        # often there are extra prefixes
        prefix = _get_key_prefix(loaded_weights.keys(), chkpt_weights.keys())
        loaded_weights = {k[len(prefix):]: v for k, v in chkpt_weights.items()}

    # HF Model class expects "model." as prefix
    # Force tensors to be contiguous
    hf_weights = {f"model.{k}": v.contiguous() for k, v in loaded_weights.items() if k != "lm_head.weight"}
    # the lm_head.weight param doesn't belong to the model submodule in HF's implementation.
    hf_weights['lm_head.weight'] = loaded_weights['lm_head.weight']
    
    tensor_filename = "model.safetensors"
    sf_filename = os.path.join("tmp", hf_repo, tensor_filename)
    dirname = os.path.dirname(sf_filename)
    os.makedirs(dirname, exist_ok=True)

    # Save safetensors
    metadata = {"format": "pt"}
    save_file(hf_weights, sf_filename, metadata=metadata)

    # Verify
    reloaded = load_file(sf_filename)
    for k in hf_weights:
        pt_tensor = hf_weights[k]
        sf_tensor = reloaded[k]
        if not torch.equal(pt_tensor, sf_tensor):
            raise RuntimeError(f"The output tensors do not match for key {k}")

    # Get config
    hf_config = _hf_config_from_model(model.config, tokenizer, next(iter(hf_weights.values())).dtype)
    config_filename = "config.json"
    with open(os.path.join("tmp", hf_repo, config_filename), "w") as f:
        json.dump(hf_config, f, indent=2)

    tokenizer.push_to_hub(hf_repo)

    # Upload
    api = HfApi()
    api.upload_folder(
        folder_path=dirname,
        path_in_repo=".",
        repo_id=hf_repo,
        repo_type="model",
        commit_message="Uploading new weights"    
    )

    shutil.rmtree("tmp")


def datatrove_tokenization_executor(hf_dataset_id,
                                    name,
                                    text_column,
                                    output_folder,
                                    tokenizer_id,
                                    eos_token,
                                    num_workers,
                                    shuffle=True,
                                    job_id=None):
    if not job_id:
        job_id = secrets.token_hex(8)

    
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
            shuffle=shuffle,
            seed=1998
        )
    ]

    executor = LocalPipelineExecutor(
        pipeline=pipeline,
        logging_dir=f"logs_{job_id}/",
        tasks=num_workers,
    )

    return executor


def main():
    hf_checkpoint = "HuggingFaceTB/SmolLM2-135M"
    tokenizer = AutoTokenizer.from_pretrained(hf_checkpoint)
    tokenizer.pad_token = tokenizer.eos_token
    
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # input_ids = tokenizer(["Gravity is", "Dark Matter is"], return_tensors="pt").to(device)['input_ids']
    # model = load_from_pretrained(hf_checkpoint).to(device)
    # idx = model.generate(input_ids, temperature=0.25, top_k=25, max_new_tokens=16)
    # print(tokenizer.batch_decode(idx))

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
    save_to_hub(model, tokenizer, "amang1802/llama_162M_fineweb10BT", checkpoint_path="fineweb-10BT/model.checkpoint.2024-12-17--11-01-13")

    # executor = datatrove_tokenization_executor(
    #     hf_dataset_id="wikimedia/wikipedia",
    #     name="20231101.hi",
    #     text_column="text",
    #     output_folder="./wiki_hindi_tok",
    #     tokenizer_id=hf_checkpoint,
    #     eos_token=tokenizer.eos_token,
    #     num_workers=16
    # )
    # executor.run()


if __name__ == "__main__":
    main()

