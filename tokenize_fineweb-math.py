from hf_utils import datatrove_tokenization_executor
from transformers import AutoTokenizer

def main():
    hf_checkpoint = "HuggingFaceTB/SmolLM-360M"
    tokenizer = AutoTokenizer.from_pretrained(hf_checkpoint)

    executor = datatrove_tokenization_executor(
        job_id="tokenize_fineweb-infiwebmath",
        hf_dataset_id="HuggingFaceTB/finemath",
        name="infiwebmath-3plus",
        id_column="url",
        text_column="text",
        output_folder="./fineweb-infiwebmath_tok",
        tokenizer_id=hf_checkpoint,
        eos_token=tokenizer.eos_token,
        shuffle=False,
        num_workers=64
    )
    executor.run()

# run as: HF_HUB_ENABLE_HF_TRANSFER=1 python tokenize_fineweb-math.py
if __name__ == "__main__":
    main()