from hf_utils import datatrove_tokenization_executor
from transformers import AutoTokenizer

def main():
    hf_checkpoint = "HuggingFaceTB/SmolLM-360M"
    tokenizer = AutoTokenizer.from_pretrained(hf_checkpoint)

    executor = datatrove_tokenization_executor(
        hf_dataset_id="wikimedia/wikipedia",
        name="20231101.es",
        id_column="id",
        text_column="text",
        output_folder="./wiki_es_tok",
        tokenizer_id=hf_checkpoint,
        eos_token=tokenizer.eos_token,
        shuffle=False,
        num_workers=16
    )
    executor.run()

# run as: HF_HUB_ENABLE_HF_TRANSFER=1 python tokenize_wikipedia.py
if __name__ == "__main__":
    main()