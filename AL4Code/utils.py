from handlers import Handler
from data import get_Code4ML
from query_strategies import LeastConfidence, MarginSampling
import csv
from nets import Net

def get_handler():
    return Handler

def get_dataset():
    return get_Code4ML(get_handler())

def get_net(device):
    return Net(device)

def get_strategy(name):
    if name == "LeastConfidence":
        return LeastConfidence
    elif name == "MarginSampling":
        return MarginSampling
    else:
        raise NotImplementedError

def save_results(results, strategy_name):

    header = ['round', 'accuracy', 'precision', 'recall', 'f1', 'f2', 'best_hp', 'training_size']

    # open the file in the write mode
    file_name = '../Results/'+strategy_name+'.csv'
        
    with open(file_name, 'w') as f:
        # create the csv writer
        writer = csv.writer(f)
        # write rows to the csv file
        writer.writerow(header)
        writer.writerows(results)
