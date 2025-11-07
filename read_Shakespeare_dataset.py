import pickle
import numpy as np
import torch
import torchvision.datasets
import torchvision.transforms as transforms
import os
import requests
from torch.utils.data import TensorDataset

def read_datasets(dataset_name, data_dir=None):
    if dataset_name in ["Shakespeare"]:
        pass
    else:
        print('New dataset, readdatasets need adjustment')
        return None, None
        

    if data_dir==None:
        data_dir = os.getcwd() + '/data/' + dataset_name + '/'
        os.makedirs(data_dir, exist_ok=True)    

    if dataset_name == "Shakespeare":
        # download the tiny shakespeare dataset
        # print(data_dir)
        input_file_path = os.path.join(data_dir, 'input.txt')
        if not os.path.exists(input_file_path):
            # print("input download started")
            data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
            response = requests.get(data_url)
            with open(input_file_path, 'w') as f:
                f.write(response.text)
            # print("input downloaded")

        with open(input_file_path, 'r') as f:
            data = f.read()

        # get all the unique characters that occur in this text
        chars = sorted(list(set(data)))
        vocab_size = len(chars)

        # create a mapping from characters to integers
        stoi = { ch:i for i,ch in enumerate(chars) }
        itos = { i:ch for i,ch in enumerate(chars) }

        def encode(s):
            return [stoi[c] for c in s] # encoder: take a string, output a list of integers
        def decode(l):
            return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

        # create the train and test splits
        n = len(data)
        train_data = data[:int(n*0.9)]
        val_data = data[int(n*0.9):]

        # encode both to integers
        train_ids = encode(train_data)
        val_ids = encode(val_data)

        # convert to tensors
        block_size = 128
        train_data = [train_ids[i:i+block_size] for i in range(0, len(train_ids) - block_size, block_size)]
        train_targets = [train_ids[i+1:i+1+block_size] for i in range(0, len(train_ids) - block_size, block_size)]
        test_data = [val_ids[i:i+block_size] for i in range(0, len(val_ids) - block_size, block_size)]
        test_targets = [val_ids[i+1:i+1+block_size] for i in range(0, len(val_ids) - block_size, block_size)]

        train_data = torch.tensor(train_data, dtype=torch.long)
        train_targets = torch.tensor(train_targets, dtype=torch.long)
        test_data = torch.tensor(test_data, dtype=torch.long)
        test_targets = torch.tensor(test_targets, dtype=torch.long)

        # wrap in TensorDataset
        train_dataset = TensorDataset(train_data, train_targets)
        test_dataset = TensorDataset(test_data, test_targets)


        # Saving meta information as well, to help us encode/decode later
        meta = {
            'vocab_size': vocab_size,
            'itos': itos,
            'stoi': stoi,
        }
        with open(os.path.join(data_dir, 'meta.pkl'), 'wb') as f:
            pickle.dump(meta, f)

        return train_dataset, test_dataset
     
