import numpy as np
import torch
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, fbeta_score
import nltk
import numpy as np
nltk.download('punkt')
from transformers import RobertaTokenizer, BertTokenizer

class Data:
        
    def __init__(self, handler, X_train, Y_train, X_test, Y_test, train_ids):

        # initialize variables
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.train_ids = train_ids
        self.handler = handler
        self.n_pool = len(X_train)
        self.n_test = len(X_test)
        
        self.labeled_idxs = np.zeros(self.n_pool, dtype=bool)

        # Load the BERT tokenizer.
        print('Loading BERT tokenizer...')
        self.tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base', do_lower_case=True)
        self.max_len = 0
        
        sentences = np.concatenate([self.X_train, self.X_test])

        # For every sentence...
        for sent in sentences:

            # Tokenize the text and add `[CLS]` and `[SEP]` tokens.
            input_ids = self.tokenizer.encode(sent, add_special_tokens=True)

            # Update the maximum sentence length.
            self.max_len = max(self.max_len, len(input_ids))

        print('Max sentence length: ', self.max_len)

        self.input_ids_train, self.attention_masks_train, self.labels_train = self.tokenize(X_train, Y_train)
        self.input_ids_test, self.attention_masks_test, self.labels_test = self.tokenize(X_test, Y_test)

    def tokenize(self, X, Y):

        # Tokenize all of the sentences and map the tokens to thier word IDs.
        input_ids = []
        attention_masks = []

        # For every sentence...
        for sent in X: 
            # `encode_plus` will:
            #   (1) Tokenize the sentence.
            #   (2) Prepend the `[CLS]` token to the start.
            #   (3) Append the `[SEP]` token to the end.
            #   (4) Map tokens to their IDs.
            #   (5) Pad or truncate the sentence to `max_length`
            #   (6) Create attention masks for [PAD] tokens.
        
            encoded_dict = self.tokenizer.encode_plus(
                                sent,                      # Sentence to encode.
                                add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                                max_length = 512, #self.max_len,           # Pad & truncate all sentences.
                                padding = "max_length",
                                truncation = True,
                                return_attention_mask = True,   # Construct attn. masks.
                                return_tensors = 'pt',     # Return pytorch tensors.
                            )
            
            # Add the encoded sentence to the list.    
            input_ids.append(encoded_dict['input_ids'])

            # And its attention mask (simply differentiates padding from non-padding).
            attention_masks.append(encoded_dict['attention_mask'])

        # Convert the lists into tensors.
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        labels = torch.tensor(Y)

        return input_ids, attention_masks, labels

    def initialize_labels(self, num, is_active_learning):
        # generate initial labeled pool
        if is_active_learning:
            tmp_idxs = np.arange(self.n_pool)
            np.random.shuffle(tmp_idxs)
            self.labeled_idxs[tmp_idxs[:num]] = True
        else:
            self.labeled_idxs[np.arange(self.n_pool)] = True
    
    def update_n_pool(self):
        self.n_pool = self.input_ids_train.size(dim=0) #len(self.X_train) 

    def get_labeled_data(self):
        labeled_idxs = np.arange(self.n_pool)[self.labeled_idxs]

        return labeled_idxs, self.X_train[labeled_idxs], self.input_ids_train[labeled_idxs], self.attention_masks_train[labeled_idxs], self.labels_train[labeled_idxs]
    
    def get_unlabeled_data(self):
        unlabeled_idxs = np.arange(self.n_pool)[~self.labeled_idxs]
        return unlabeled_idxs, self.input_ids_train[unlabeled_idxs], self.attention_masks_train[unlabeled_idxs], self.labels_train[unlabeled_idxs]

    def get_test_data(self):
        return self.X_test, self.input_ids_test, self.attention_masks_test, self.labels_test
    
    def cal_test_acc(self, preds):
        return 1.0 * np.sum(self.Y_test == preds) / self.n_test
    
    def cal_test_precision(self, preds):
        return precision_score(self.Y_test, preds)
    
    def cal_test_recall(self, preds):
        return recall_score(self.Y_test, preds)
    
    def cal_test_f1(self, preds):
        return f1_score(self.Y_test, preds)
    
    def cal_test_f2(self, preds):
        return fbeta_score(self.Y_test, preds, beta=2)


def get_Code4ML(handler):

    train_data = pd.read_csv("../Dataset/train_annotated.csv")
    test_data = pd.read_csv("../Dataset/test_annotated.csv")

    train_data = train_data.dropna()
    test_data = test_data.dropna()

    train_data['code_block'] = train_data['code_block'].str.lower()
    test_data['code_block'] = test_data['code_block'].str.lower()

    X_train = train_data['code_block'].values
    X_test = test_data['code_block'].values

    y_train = train_data['has_data_leakage'].values
    y_test = test_data['has_data_leakage'].values

    train_ids = train_data['id'].values

    return Data(handler, X_train, y_train, X_test, y_test, train_ids)

