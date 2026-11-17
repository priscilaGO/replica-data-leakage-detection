import torch
from collections import Counter
import numpy as np
import os
from openai import OpenAI  # PATCH: openai>=1.0 client

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

_client = None  # PATCH: created lazily so runs without augmentation do not need an API key
AUG_MODEL = os.environ.get("AUG_MODEL", "gpt-3.5-turbo-1106")  # model used in the paper

# ---------------------------------------------------------------------------------------------
# PATCH: backend local (LLM abierto) para el aumento de datos, activado con AUG_BACKEND=local.
# Reemplaza a gpt-3.5-turbo-1106 cuando no hay API key de OpenAI (DESVIACION respecto al paper).
# ---------------------------------------------------------------------------------------------
class _Msg:
    def __init__(self, c): self.content = c
class _Choice:
    def __init__(self, c): self.message = _Msg(c)
class _Resp:
    def __init__(self, texts): self.choices = [_Choice(t) for t in texts]

_local = {}

def _llm_to(device):
    """Mueve el LLM local a la GPU solo mientras se aumenta, para no competir por memoria con CodeBERT."""
    if os.environ.get("AUG_BACKEND") != "local" or "model" not in _local:
        return
    _local["model"].to(device)
    if device == "cpu":
        torch.cuda.empty_cache()

def _local_completion(messages, n=1, **_):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    if not _local:
        name = os.environ.get("LOCAL_LLM", "Qwen/Qwen2.5-Coder-3B-Instruct")
        _local["tok"] = AutoTokenizer.from_pretrained(name)
        _local["model"] = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16, low_cpu_mem_usage=True)
        _local["model"].to("cuda")
        _local["model"].eval()
    tok, model = _local["tok"], _local["model"]
    msgs = [dict(m) for m in messages]
    ids = tok(msgs[-1]["content"], add_special_tokens=False).input_ids
    if len(ids) > 1500:  # codigos muy largos se cortan
        msgs[-1]["content"] = tok.decode(ids[:1500])
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, do_sample=True, temperature=0.7, top_p=0.95, max_new_tokens=1024,
                             num_return_sequences=n, pad_token_id=tok.eos_token_id)
    return _Resp([tok.decode(o[inputs.input_ids.shape[1]:], skip_special_tokens=True) for o in out])


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs):
    if os.environ.get("AUG_BACKEND") == "local":
        return _local_completion(**kwargs)
    global _client
    if _client is None:
        _client = OpenAI()  # reads OPENAI_API_KEY from the environment
    return _client.chat.completions.create(**kwargs)

class Strategy:

    def __init__(self, dataset, net):
        self.dataset = dataset
        self.net = net

    def query(self, n):
        pass
    
    def update(self, pos_idxs, neg_idxs=None):
        self.dataset.labeled_idxs[pos_idxs] = True
        if neg_idxs:
            self.dataset.labeled_idxs[neg_idxs] = False

    def augement(self, pos_idxs):
        _llm_to("cuda")
        try:
            return self._augement(pos_idxs)
        finally:
            _llm_to("cpu")

    def _augement(self, pos_idxs):

        X = self.dataset.X_train[pos_idxs]
        y = self.dataset.Y_train[pos_idxs]
        counter = Counter(y)

        no_augmented_samples = 0  # PATCH: original crashes if no positive is labeled yet
        if 1 in counter:
            positives = counter[1]
            negatives = counter[0]
            diff = negatives/positives
            no_augmented_samples = int(round(diff,0) - 1)

        augmented_X = []
        augmented_y = []
        
        #GPT
        if no_augmented_samples != 0:
            for idx, x in enumerate(X):

                if y[idx] == 0: # 0 is the majority class
                    continue
                
                print("augmenting one sample...")
                messages = [
                {
                    "role": "system", 
                    "content": '''
                    You are a specialist in machine learning. 
                    You will be provided with a piece of Python code, and your task is to refactor this code. 
                    Rrandomly use a combination of some of the refactoring techinques: Variable Renaming, 
                    Variable Extraction, Change Function Declaration, Change Reference to Value, Inline Function, 
                    or Introduce Special Case.  
                    Omit the import statements.'''
                    },
                {
                    "role": "user", 
                    "content": x
                    }
                ]

                completion = completion_with_backoff(
                    model=AUG_MODEL, 
                    messages=messages,
                    n = no_augmented_samples
                )

                for choice in completion.choices:
                    augmented_X.append(choice.message.content)
                    augmented_y.append(y[idx])

            # tokenize augmented data 
            augmented_input_ids_train, augmented_attention_masks_train, augmented_labels_train = self.dataset.tokenize(augmented_X, augmented_y)

            # add the augemnted sentences to the training data
            self.dataset.input_ids_train = torch.cat((self.dataset.input_ids_train, augmented_input_ids_train), 0)
            self.dataset.attention_masks_train = torch.cat((self.dataset.attention_masks_train, augmented_attention_masks_train), 0)
            self.dataset.labels_train = torch.cat((self.dataset.labels_train, augmented_labels_train), 0)
            self.dataset.X_train = np.append(self.dataset.X_train, augmented_X)
            self.dataset.Y_train = np.append(self.dataset.Y_train, augmented_y)

            # update labeled_idxs
            new_labeled_idxs = np.full(len(augmented_y), True)
            labeled_idxs = np.concatenate((self.dataset.labeled_idxs, new_labeled_idxs))
            
            # update number of samples in the pool
            self.dataset.update_n_pool()
            self.dataset.labeled_idxs = np.zeros(self.dataset.n_pool, dtype=bool)
            self.dataset.labeled_idxs[labeled_idxs] = True
        else:
            print("No augumentation is needed.")

    def train_bert(self):

        # load data
        _, _, input_ids, attention_masks, labels = self.dataset.get_labeled_data()

        # train
        _ = self.net.train_bert(input_ids, attention_masks, labels, 5e-5, 16, 4)
        return [4, 5e-5, 16], len(labels)

    def predict_bert(self, input_ids, attention_masks, labels):
        preds = self.net.predict_bert(input_ids, attention_masks, labels)
        return preds
    
    def predict_prop_bert(self, input_ids, attention_masks, labels):
        preds = self.net.predict_prop_bert(input_ids, attention_masks, labels)
        return preds

