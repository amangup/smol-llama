from torch.nn.utils.rnn import pad_sequence
from typing import Dict, List, Optional, Tuple, Type, TypeVar, Union
from tqdm import tqdm

import lm_eval
import torch
import torch.nn.functional as F

class SmolEvalWrapper(lm_eval.api.model.TemplateLM):
    def __init__(self, model, tokenizer, device, batch_size=1, max_seq_len=1024):
        super().__init__()
        self._model = model
        self._tokenizer = tokenizer
        self._device = device
        self._batch_size = 1
        self._max_seq_length = max_seq_len
        self._max_new_tokens = 128
        self._log_softmax = torch.nn.LogSoftmax(dim=1)

    @property
    def eot_token_id(self):
        return self._tokenizer.eos_token_id()

    @property
    def prefix_token_id(self):
        return self._tokenizer.bos_token_id()

    @property
    def max_length(self):
        return self._max_seq_length

    @property
    def max_gen_toks(self):
        return self._max_new_tokens

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device(self):
        return self._device

    def tok_encode(self, string: str, **kwargs):
        output = self._tokenizer(string)
        return output['input_ids']

    def tok_decode(self, tokens):
        return self._tokenizer.decode(tokens)

    def _model_call(self, x):
        logits, _, _ = self._model(x)
        return logits

    def _model_generate(self, contexts, until):
        pass
        # input_ids = self._tokenizer(contexts, return_tensors='pt', padding="longest", padding_side='left')['input_ids'].to(self._device)
        #
        # outputs = [""] * len(contexts)
        # for _ in range(max_new_tokens):
        #     logits, _ = self(idx)
        #     logits = logits[:, -1, :]
        #
        #     idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        #     idx_new = torch.cat((idx, idx_next), dim=1)
        #
        #     if stop:
        #         last_few_tokens = idx_new[:, -5:]
        #         strs = tokenizer.batch_decode(last_few_tokens)
        #
        #         for seq_i in range(idx.size(0)):
        #             # find if stop seq matches
        #             # remove and get the clean string
        #             # set it to the right position in output
        #             # remove the seq from generation
        #             #
        #             # any((re.search(f"{stop_str}$", strs[seq_i]) for stop_str in stop))):
        #             # outputs[i] =
        #
        #     idx = idx_new
        #
        #     if torch.all(idx[:, -1] == tokenizer.eos_token_id):
        #         break
        #
        # # decode those which ended due reaching max length
        # # output_strs = self._tokenizer.batch_decode(outputs, skip_special_tokens=True)
        #
        # return output_strs
        

    def generate_until(self, requests) -> List[str]:
        """Generate greedily until a stopping sequence

        :param requests: list[Instance]
            A list of Instance objects with property `args` which returns a tuple (context, gen_kwargs).
            context: str
                Context string
            gen_kwargs: dict
                A dictionary of keyword arguments to pass to the generation function e.g. top_k, until, etc.
        :return: list[str]
            A list of model generated continuations.
            continuation: str
                The generated continuation.
        """
        outputs = []
        for st_i in tqdm(range(0, len(requests), self._batch_size)):
            batch = requests[st_i: st_i + self._batch_size]
            contexts = [req.args[0] for req in batch]
            stop = batch[0].args[1]['until']
            
            output_batch = self._model_generate(contexts, stop)

            outputs.extend(output_batch)

        return outputs
            

    def _loglikelihood_tokens(self, requests, **kwargs) -> List[Tuple[float, bool]]:
        """Generate loglikelihood of completions given contexts

        :param requests: List[Tuple[Tuple[str, str], List[int], List[int]]]
            List[Tuple[Tuple[context_str, completion_str], context_tokens, completion_tokens]]

        :return: List[Tuple[float, bool]]
            A list of sum_logprobs, exact_match
        """
        outputs = []
        for st_i in tqdm(range(0, len(requests), self._batch_size)):
            batch = requests[st_i: st_i + self._batch_size]
            
            # Inputs pytorch tensor
            inputs_n_lengths = [(context+cont, len(context), len(cont)) for _, context, cont in batch]
            inputs, context_lengths, completion_lengths = [list(x) for x in zip(*inputs_n_lengths)]
            
            tensors = [torch.tensor(seq) for seq in inputs]
            input_tensor = pad_sequence(tensors, batch_first=True, padding_value=self._tokenizer.pad_token_id, padding_side='right')
            input_tensor = input_tensor[:, :self._max_seq_length]
                    
            X = input_tensor[:, :-1].to(self._device)
            logits = self._model_call(X)
    
            # Sum logprobs
            Y = input_tensor[:, 1:].to(self._device)
            
            ## Set ignored output tokens to be -1
            completion_tokens_mask = torch.zeros(Y.shape).to(self._device)
            for i, (start, num_tokens) in enumerate(zip(context_lengths, completion_lengths)):
                completion_tokens_mask[i, (start-1):(start+num_tokens-1)] = 1
            Y[completion_tokens_mask == 0] = -1
    
            ## Cross entropy
            seq_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), Y.reshape(-1), ignore_index=-1, reduction='none')
            seq_loss = seq_loss.reshape(X.size(0), -1).sum(dim=-1)
        
            # Exact match
            greedy_completion = torch.argmax(logits, dim=-1)
            greedy_completion[completion_tokens_mask == 0] = -1
            exact_matches = torch.all(Y == greedy_completion, dim=1)
    
            output = [(-sum_logprobs.item(), match.item()) for sum_logprobs, match in zip(seq_loss, exact_matches)]
            outputs.extend(output)

        return outputs
        

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False) -> List[float]:
        pass


@torch.no_grad()
def run_benchmarks(model_eval_wrapper, tasks, limit=None):

    try:
        lm_eval.tasks.initialize_tasks()
    except:
        pass

    task_dict = lm_eval.tasks.get_task_dict(tasks)
        
    eval_results = lm_eval.evaluator.evaluate(
        model_eval_wrapper,
        task_dict,
        limit=limit
    )
    
    return eval_results