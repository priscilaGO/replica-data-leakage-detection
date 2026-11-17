import argparse
import numpy as np
import torch
from utils import get_dataset, get_net, get_strategy, save_results
from pprint import pprint
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--n_init_labeled', type=int, default=20, help="number of init labeled samples")
parser.add_argument('--n_query', type=int, default=20, help="number of queries per round")
parser.add_argument('--strategy_name', type=str, default="LeastConfidence", choices=[ "LeastConfidence", "MarginSampling"], help="query strategy")
parser.add_argument('--patient', type=int, default=5, help="number of round with no improvement after which active learning will be stopped")
parser.add_argument('--active_learning', action='store_true', help="enable or disable active learning")
parser.add_argument('--seed', type=int, default=1, help="random seed (PATCH: added)")
parser.add_argument('--tag', type=str, default="", help="suffix for the results file (PATCH: added)")
parser.add_argument('--with_augmentation', action='store_true', help="enable or disable augmentation")

args = parser.parse_args()
pprint(vars(args))
print()

# PATCH: seeds (the original only fixed a DataLoader generator)
import random
random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

# device
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

dataset = get_dataset() # load dataset
net = get_net(device) # load network
strategy = get_strategy(args.strategy_name)(dataset, net) # load strategy

# start experiment
dataset.initialize_labels(args.n_init_labeled, args.active_learning)
print(f"number of labeled pool: {args.n_init_labeled}")
print(f"number of unlabeled pool: {dataset.n_pool-args.n_init_labeled}")
print(f"number of testing pool: {dataset.n_test}")
print()

PATIENT = args.patient
results = []

# ******* start round 0 *********
print("Round 0")

if args.with_augmentation:
    strategy.augement(dataset.labeled_idxs)

# train
best_hp, training_size = strategy.train_bert()

# predict
X_test, input_ids_test, attention_masks_test, labels_test = dataset.get_test_data()
preds = strategy.predict_bert(input_ids_test, attention_masks_test, labels_test)

# evaluate
acc = dataset.cal_test_acc(preds)
precision = dataset.cal_test_precision(preds)
recall = dataset.cal_test_recall(preds)
f1 = dataset.cal_test_f1(preds)
f2 = dataset.cal_test_f2(preds)

print(f"Round 0 testing accuracy: {acc}")
print(f"Round 0 testing precision: {precision}")
print(f"Round 0 testing recall: {recall}")
print(f"Round 0 testing f1: {f1}")
print(f"Round 0 testing f2: {f2}")
results.append([0, acc, precision, recall, f1, f2, best_hp, training_size])

# ******* end round 0 *********

# ******* start next rounds *********
if args.active_learning:

    rd = 1
    patinet = PATIENT
    while True:

        print(f"Round {rd}")

        # query
        query_idxs = strategy.query(args.n_query)

        # update labels
        strategy.update(query_idxs)

        if args.with_augmentation:
            strategy.augement(dataset.labeled_idxs)

        # train
        best_hp, training_size = strategy.train_bert()
        
        # predict
        preds = strategy.predict_bert(input_ids_test, attention_masks_test, labels_test)

        # evaluate
        best_f2_so_far = max(r[5] for r in results)  # PATCH: np.array(results) is ragged (best_hp is a list)
        acc = dataset.cal_test_acc(preds)
        precision = dataset.cal_test_precision(preds)
        recall = dataset.cal_test_recall(preds)
        f1 = dataset.cal_test_f1(preds)
        f2 = dataset.cal_test_f2(preds)
        print(f"Round {rd} testing accuracy: {acc}")
        print(f"Round {rd} testing precision: {precision}")
        print(f"Round {rd} testing recall: {recall}")
        print(f"Round {rd} testing f1: {f1}")
        print(f"Round {rd} testing f2: {f2}")
        results.append([rd, acc, precision, recall, f1, f2, best_hp, training_size])

        rd =  rd + 1

        # early stopping
        if round(results[-1][5], 3) <= round(best_f2_so_far, 3):
            patinet = patinet - 1
            if patinet == 0:
                break
        else:
            patinet = PATIENT

print("Done")
# save results
save_results(results, args.strategy_name + ("_" + args.tag if args.tag else ""))




