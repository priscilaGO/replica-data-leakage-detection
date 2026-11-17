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

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs):
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

