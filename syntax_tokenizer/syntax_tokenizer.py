from collections import defaultdict
from datasets import load_dataset
from tqdm import tqdm

import bisect
import math
import numpy as np
import pickle

import spacy
from spacy_syllables import SpacySyllables


class SyntaxTokenizer:
    def __init__(self, spacy_model="en_core_web_sm", lang="en_US"):
        self.nlp = spacy.load(spacy_model, exclude=['ner'])
        self.nlp.add_pipe("syllables", after="tagger", config={"lang": lang})
        #print(self.nlp.pipeline)


    def train(self, texts):
        self.pos_to_int = {}
        for i, label in enumerate(self.nlp.get_pipe("tagger").labels):
            #print(label, " -- ", spacy.explain(label))
            self.pos_to_int[label] = i + 1

        #print(self.pos_to_int)

        self.max_syllables = 0
        self.idf = defaultdict(int)

        for doc in tqdm(self.nlp.pipe(texts, n_process=16)):
            unique_tokens = set()
            for token in doc:
                self.max_syllables = max(self.max_syllables, token._.syllables_count or 0)
                unique_tokens.add(token.lemma)
                #print(f"Text: {token.text}; syllables: { token._.syllables}, {token._.syllables_count}")

            for token in unique_tokens:
                self.idf[token] += 1

#        print(self.idf)

        n = len(texts)
        self.idf = {k: math.log(n/v) for k, v in self.idf.items()}
        idf_vals = np.array(list(self.idf.values()), dtype=np.float32)
        # Discretize idf values using dynamic bin edges
        self.bin_edges = np.histogram_bin_edges(idf_vals, bins='sturges')

        # to allow for None, and some uniquely complex words not in training set
        self.max_syllables = self.max_syllables + 2

        #print(self.max_syllables)

    def save(self, path):
        pickle.dump(self, open(path, "wb"))

    @staticmethod
    def load(path):
        return pickle.load(open(path, "rb"))

    def vocab_size(self):
        return (self.max_syllables +
                (self.max_syllables+1) * len(self.bin_edges) +
                (self.max_syllables + 1) * (len(self.bin_edges) + 1) * len(self.pos_to_int)
               )

    def encode(self, doc, return_str=False):
        encoding = []
        for token in self.nlp(doc):
            num_syllables = token._.syllables_count
            if not num_syllables:
                num_syllables = 0
            num_syllables = min(num_syllables, self.max_syllables)

            idf = self.idf.get(token.text, 0.001)
            idf_bin = bisect.bisect_left(self.bin_edges, idf)

            token_val = (num_syllables +
                         (self.max_syllables+1) * idf_bin +
                         (self.max_syllables+1) * (len(self.bin_edges)+1) *  self.pos_to_int[token.tag_]
                        )

            if return_str:
                encoding.append((token.text, token_val))
            else:
                encoding.append(token_val)

        return encoding

def main():
    # load docs dataset
    docs_ds = load_dataset('amang1802/synthetic_data_topic_conditioned_L3.3_70B')['train']
    docs = docs_ds['text']

    # run train
    tokenizer = SyntaxTokenizer()
    tokenizer.train(docs)

    print(f"Vocab size: {tokenizer.vocab_size()}")

    text1 = "South Korea's Constitutional Court removes Yoon Suk Yeol (pictured) as the president of South Korea, following his declaration of martial law."
    text2 = "US president Donald Trump announces trade tariffs on most countries."
    text3 = "Marine Le Pen, the runner-up in the 2017 and 2022 French presidential elections, is convicted of embezzlement and banned from standing in elections for five years."
    text4 = "A magnitude-7.7 earthquake leaves more than 4,300 people dead in Myanmar and Thailand."

    # encode some sample texts
    print(tokenizer.encode(text1, return_str=True))
    print(tokenizer.encode(text2, return_str=True))
    print(tokenizer.encode(text3, return_str=True))
    print(tokenizer.encode(text4, return_str=True))

    tokenizer.save("tokenizer.data")
    tok2 = SyntaxTokenizer.load("tokenizer.data")

    print(tok2.encode(text1, return_str=True))


if __name__ == '__main__':
    main()