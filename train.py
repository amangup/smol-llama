from dataclasses import dataclass
from transformers import AutoTokenizer

import math
import torch


@dataclass
class TrainerConfig:
    learning_rate: float = 1e-3
    per_device_train_batch_size: int = 32
    max_seq_len: int = 2048
    num_epochs: int = 1


class DataLoader:
    def __init__(self, config, tokenizer, text):
        self.config = config
        self.tokenizer = tokenizer
        self.tokens = self.tokenize(text)
        self.index = 0
        self.batch_tensor_shape = (config.per_device_train_batch_size, config.max_seq_len)


    def tokenize(self, text):
        outputs = self.tokenizer(
            text,
            max_length=self.config.max_seq_len + 1,
            padding="max_length",
            truncation=True,
            return_overflowing_tokens=True,
            return_tensors="pt",
            stride=1
        )

        return outputs['input_ids']


    def next_batch(self):
        new_index = self.index + self.config.per_device_train_batch_size
        x = self.tokens[self.index:new_index, :-1]
        y = self.tokens[self.index:new_index, 1:]

        self.index = new_index
        if new_index > self.tokens.size(0):
            self.index = 0

        return x, y


    def num_steps_per_epoch(self):
        return math.ceil(self.tokens.size(0) / self.config.per_device_train_batch_size)


class Trainer:
    def __init__(self, config, model):
        self.config = config
        self.model = model
        self.device = model.device

    def train(self, train_dataloader, eval_dataloader):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate)
        for epoch in range(self.config.num_epochs):
            for step in range(train_dataloader.num_steps_per_epoch()):
                optimizer.zero_grad()

                X, Y = train_dataloader.next_batch()
                _, loss = self.model(X, Y)

                print(f"Step: {step}, Training Loss: {loss.item()}")

                loss.backward()
                optimizer.step()




def main():
    hf_checkpoint = "HuggingFaceTB/SmolLM2-135M"
    tokenizer = AutoTokenizer.from_pretrained(hf_checkpoint)
    tokenizer.pad_token = tokenizer.eos_token

    print(type(tokenizer))

    config = TrainerConfig(
        per_device_train_batch_size=8,
        max_seq_len=16,
    )

    text = "Calm. Kindness. Kinship. Love. I've given up all chance at inner peace. I've made my mind a sunless space. I share my dreams with ghosts. I wake up every day to an equation I wrote 15 years ago from which there's only one conclusion, I'm damned for what I do. My anger, my ego, my unwillingness to yield, my eagerness to fight, they've set me on a path from which there is no escape. I yearned to be a savior against injustice without contemplating the cost and by the time I looked down there was no longer any ground beneath my feet. What is my sacrifice? I'm condemned to use the tools of my enemy to defeat them. I burn my decency for someone else's future. I burn my life to make a sunrise that I know I'll never see. And the ego that started this fight will never have a mirror or an audience or the light of gratitude. So what do I sacrifice? Everything! You'll stay with me, Lonni. I need all the heroes I can get."
    dataloader = DataLoader(config, tokenizer, text)
    for i in range(4):
        x, y = dataloader.next_batch()
        print(f"X: {x.shape} -", x)
        print(f"Y: {y.shape} -", y)


if __name__ == "__main__":
    main()
