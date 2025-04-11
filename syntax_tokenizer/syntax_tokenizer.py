from collections import defaultdict
from dataclasses import dataclass
from tqdm import tqdm

import bisect
import math
import numpy as np
import pickle
import torch

import spacy
from spacy_syllables import SpacySyllables

@dataclass
class SyntaxTokenizerData:
    spacy_model: str
    lang: str

    eos_token: str
    eos_token_id: int

    pad_token: str
    pad_token_id: int

    mask_token: str
    mask_token_id: int

    special_tokens: dict[str, int]
    num_special_tokens: int

    pos_to_int: dict[str, int]
    int_to_pos: dict[int, str]

    idf: dict[str, float]
    bin_edges: np.ndarray[np.float64]

    max_syllables: int


class SyntaxTokenizer:
    def __init__(self, spacy_model="en_core_web_sm", lang="en_US"):
        self.spacy_model = spacy_model
        self.lang = lang

        self.nlp = spacy.load(spacy_model, exclude=['ner'])
        self.nlp.add_pipe("syllables", after="tagger", config={"lang": lang})


    def train(self, texts):
        pos_to_int = {}
        int_to_pos = {}
        for i, label in enumerate(self.nlp.get_pipe("tagger").labels):
            #print(label, " -- ", spacy.explain(label))
            pos_to_int[label] = i + 1
            int_to_pos[i + 1] = label


        max_syllables = 0
        idf = defaultdict(int)

        for doc in tqdm(self.nlp.pipe(texts, n_process=16)):
            unique_tokens = set()
            for token in doc:
                max_syllables = max(max_syllables, token._.syllables_count or 0)
                unique_tokens.add(token.lemma_)
                #print(f"Text: {token.lemma_}; syllables: { token._.syllables}, {token._.syllables_count}")

            for token in unique_tokens:
                idf[token] += 1

        #print(idf)

        n = len(texts)
        idf = {k: math.log(n/v) for k, v in idf.items()}
        idf_vals = np.array(list(idf.values()), dtype=np.float32)
        # Discretize idf values using dynamic bin edges
        bin_edges = np.histogram_bin_edges(idf_vals, bins='sturges')

        # to allow for None, and some uniquely complex words not in training set
        max_syllables = max_syllables + 2
        #print(max_syllables)

        special_tokens = {
            "</s>": 0,
            "<mask>": 1
        }

        self.data = SyntaxTokenizerData(
            spacy_model=self.spacy_model,
            lang=self.lang,
            eos_token="</s>",
            eos_token_id=0,
            pad_token="</s>",
            pad_token_id=0,
            mask_token="<mask>",
            mask_token_id=1,
            special_tokens=special_tokens,
            num_special_tokens=len(special_tokens),
            pos_to_int=pos_to_int,
            int_to_pos=int_to_pos,
            idf=idf,
            bin_edges=bin_edges,
            max_syllables=max_syllables,
        )

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.data, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            tokenizer_data = pickle.load(f)

        tokenizer = SyntaxTokenizer()
        tokenizer.data = tokenizer_data

        return tokenizer

    def get_added_vocab(self):
        return self.data.special_tokens

    @property
    def vocab_size(self):
        return self.data.num_special_tokens + (self.data.max_syllables +
                (self.data.max_syllables + 1) * len(self.data.bin_edges) +
                (self.data.max_syllables + 1) * (len(self.data.bin_edges) + 1) * len(self.data.pos_to_int)
               )

    # return value --> { "input_ids": <>, "attention_mask": <> }
    def __call__(self,
                 texts,
                 max_length=-1,
                 padding="max_length",
                 truncation=False,
                 return_overflowing_tokens=False,
                 stride=0,
                 return_tensors="py"):
        if not isinstance(texts, list):
            texts = [texts]

        input_ids = []
        for text in texts:
            token_ids = self.encode(text, max_length, truncation, return_overflowing_tokens, stride)
            input_ids.extend(token_ids)

        if return_tensors != "py" and (padding == "do_not_pad" or not padding):
            padding = "longest"

        pad_to_length = 0
        match padding:
            case True | "longest":
                pad_to_length = max(len(id_seq) for id_seq in input_ids)
            case "max_length":
                pad_to_length = max_length

        attention_masks = []
        for seq in input_ids:
            k = len(seq)
            pad_tokens = max(pad_to_length - k, 0)
            seq.extend([self.data.pad_token_id] * pad_tokens)
            attention_masks.append([1] * k + [0] * pad_tokens)

        match return_tensors:
            case "pt":
                dtype = torch.int64
                input_ids = torch.tensor(input_ids, dtype=dtype)
                attention_masks = torch.tensor(attention_masks, dtype=dtype)
            case "np":
                dtype = np.int64
                input_ids = np.array(input_ids, dtype=dtype)
                attention_masks = np.array(attention_masks, dtype=dtype)

        return {"input_ids": input_ids, "attention_mask": attention_masks}


    # return value --> [[token seq]]
    def encode(self,
               text,
               max_length=-1,
               truncation=False,
               return_overflowing_tokens=False,
               stride=0):

        encoding = []
        for token in self.nlp(text):
            num_syllables = token._.syllables_count
            if not num_syllables:
                num_syllables = 0
            num_syllables = min(num_syllables, self.data.max_syllables)

            idf = self.data.idf.get(token.lemma_, 0.001)
            idf_bin = bisect.bisect_left(self.data.bin_edges, idf)
            token_val = self.data.num_special_tokens + (num_syllables +
                         (self.data.max_syllables+1) * idf_bin +
                         (self.data.max_syllables+1) * (len(self.data.bin_edges)+1) * self.data.pos_to_int[token.tag_]
                        )


            encoding.append(token_val)

        encodings = [encoding]
        if max_length > 0:
            if return_overflowing_tokens:
                assert stride < max_length
                n = len(encoding)
                encodings = [encoding[i:i+max_length] for i in range(0, n-stride, max_length-stride)]
            elif truncation:
                del encoding[max_length:]

        return encodings


    def decode(self, encoding):
        decoded = []
        for token_id in encoding:
            token_id = int(token_id)
            if token_id == self.data.eos_token_id:
                decoded.append(self.data.eos_token)
            elif token_id == self.data.pad_token_id:
                decoded.append(self.data.pad_token)
            else:
                token_id = token_id - self.data.num_special_tokens

                num_syllables = token_id % (self.data.max_syllables+1)

                token_id = token_id // (self.data.max_syllables+1)
                idf_bin = token_id % (len(self.data.bin_edges)+1)

                token_id = token_id // (len(self.data.bin_edges)+1)
                pos = self.data.int_to_pos[token_id]

                decoded.append((pos, idf_bin, num_syllables))

        return decoded

    def batch_decode(self, encodings):
        return [self.decode(encoding) for encoding in encodings]
