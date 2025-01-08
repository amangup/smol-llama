from hf_utils import datatrove_tokenization_executor
from transformers import AutoTokenizer

def main():
    hf_checkpoint = "HuggingFaceTB/SmolLM2-135M"
    tokenizer = AutoTokenizer.from_pretrained(hf_checkpoint)

    executor = datatrove_tokenization_executor(
        job_id="tokenize_fineweb-edu_sample100BT",
        hf_dataset_id="HuggingFaceFW/fineweb-edu",
        name="sample-100BT",
        text_column="text",
        output_folder="./fineweb-edu_tok-100BT",
        tokenizer_id=hf_checkpoint,
        eos_token=tokenizer.eos_token,
        shuffle=False,
        num_workers=32
    )
    executor.run()

# run as: HF_HUB_ENABLE_HF_TRANSFER=1 python tokenize_fineweb.py
if __name__ == "__main__":
    main()