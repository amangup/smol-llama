from collections import defaultdict
from dataclasses import dataclass
from tqdm import tqdm

import bisect
import math
import numpy as np
import pickle
import torch

import torch.nn.utils.rnn as R
import torch.nn.functional as F

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


SYLLABLES_UPPER_LIMIT = 20


class SyntaxTokenizer:
    def __init__(self, spacy_model="en_core_web_sm", lang="en_US"):
        self.spacy_model = spacy_model
        self.lang = lang

        #spacy.prefer_gpu()
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

        for doc in tqdm(self.nlp.pipe(texts, n_process=4)):
            unique_tokens = set()
            for token in doc:
                max_syllables = max(max_syllables, token._.syllables_count or 0)
                unique_tokens.add(token.lemma_)
                #print(f"Text: {token.lemma_}; syllables: { token._.syllables}, {token._.syllables_count}")

            for token in unique_tokens:
                idf[token] += 1

        #print(idf)

        max_syllables = min(max_syllables, 18)

        n = len(texts)
        idf = {k: math.log(n/v) for k, v in idf.items()}
        idf_vals = np.array(list(idf.values()), dtype=np.float32)
        # Discretize idf values using dynamic bin edges
        bin_edges = np.histogram_bin_edges(idf_vals, bins='sturges')

        # to allow for None, and some uniquely complex words not in training set
        max_syllables = min(max_syllables + 2, SYLLABLES_UPPER_LIMIT)
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

    @property
    def mask_token_id(self):
        return self.data.mask_token_id

    # In order of priority
    # return_overflowing_tokens=True -> return tensor of shape (B, max_length), where B >= len(texts), and padding is added where needed
    # else, if truncation=True -> return tensor of shape (len(texts), max_length)
    # else, if padding="max_length", return tensor of shape (len(texts), max_length), and padding is only added if max_length > longest_length
    # else, if padding="longest" -> return tensor of shape (len(texts), longest_length)
    def __call__(self,
                 texts,
                 max_length=-1,
                 padding="longest",
                 truncation=False,
                 return_overflowing_tokens=False,
                 stride=0,
                 return_tensors="pt",
                 nproc=8):
        if not isinstance(texts, list):
            texts = [texts]

        doc_tensors = []
        for doc in tqdm(self.nlp.pipe(texts, n_process=nproc)):
            num_syllables = []
            idf_bins = []
            pos_ids = []
            for token in doc:
                num_syllables.append(token._.syllables_count or 0)
                idf = self.data.idf.get(token.lemma_, 0.001)
                idf_bins.append(bisect.bisect_left(self.data.bin_edges, idf))

                pos_ids.append(self.data.pos_to_int[token.tag_])

            doc_raw = torch.stack([torch.tensor(num_syllables, dtype=torch.int32),
                         torch.tensor(idf_bins, dtype=torch.int32),
                         torch.tensor(pos_ids, dtype=torch.int32),
                         ], dim=0) # shape (C, L)

            if max_length > 0 and (return_overflowing_tokens or truncation):
                if return_overflowing_tokens:
                    assert stride < max_length
                    num_seqs = math.ceil(doc_raw.shape[1] / (max_length - stride))
                    pad_length = num_seqs * max_length - doc_raw.shape[1]
                    padded_doc = F.pad(doc_raw, (0, pad_length), value=-100)
                    print(padded_doc.shape)

                    doc_shaped = padded_doc.unfold(-1, size=max_length, step=(max_length-stride)).transpose(0,1) # shape (B, C, max_length)
                elif truncation:
                    if max_length > doc_raw.shape[-1]:
                        doc_shaped = F.pad(doc_raw, (0, max_length - doc_raw.shape[1]), value=-100).unsqueeze(0)
                    else:
                        doc_shaped = doc_raw[:, :max_length].unsqueeze(0)
            else:
                doc_shaped = doc_raw # shape (C, actual_length)

            doc_tensors.append(doc_shaped)


        print([doc_tensor.shape for doc_tensor in doc_tensors])

        if max_length > 0 and (return_overflowing_tokens or truncation):
            # Expect shape (B, max_length, C) for each element in doc_tensor
            batched_channels = torch.cat(doc_tensors, dim=0)
        else:
            # Expect shape (C, actual_length) for each element in doc_tensor
            # pad_sequence expects shape (actual_length, *), so we need to do tranposes
            batched_channels = R.pad_sequence([doc_tensor.transpose(0, 1) for doc_tensor in doc_tensors],
                                              batch_first=True, padding_value=-100, padding_side='right').transpose(1, 2)
            if max_length > batched_channels.shape[-1] and padding == "max_length":
                batched_channels = F.pad(batched_channels, (0, max_length-batched_channels.shape[-1]), value=-100)



        input_ids = (
            (self.data.num_special_tokens +
             (batched_channels[:, 0, :]) +
             (batched_channels[:, 1, :] * (self.data.max_syllables + 1)) +
             (batched_channels[:, 2, :] * (self.data.max_syllables + 1) * (len(self.data.bin_edges) + 1)))
            .view(-1, batched_channels.shape[-1])
        )

        input_ids[(batched_channels[:, 0, :]  == -100)] = self.data.pad_token_id

        attention_mask = (input_ids != 0).to(input_ids.dtype)

        if return_tensors == "np":
            input_ids = input_ids.numpy()
            attention_mask = attention_mask.numpy()
        elif return_tensors == "py":
            input_ids = input_ids.tolist()
            attention_mask = attention_mask.tolist()

        return {"input_ids": input_ids, "attention_mask": attention_mask}


    # return value --> [[token seq]]
    def encode(self,
               text,
               max_length=-1,
               truncation=False):

        response = self(text, max_length=max_length, truncation=truncation, return_tensors="py")
        return response["input_ids"]


    def decode(self, encoding):
        decoded = []
        for token_id in encoding:
            token_id = int(token_id)
            # TODO: generalize this
            if token_id == self.data.eos_token_id:
                decoded.append(self.data.eos_token)
            elif token_id == self.data.pad_token_id:
                decoded.append(self.data.pad_token)
            elif token_id == self.data.mask_token_id:
                decoded.append(self.data.mask_token)
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
