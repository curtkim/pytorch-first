# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import argparse

parser = argparse.ArgumentParser(description='PyTorch MNIST Example') 
parser.add_argument('--batch-size', type=int, default=64, metavar='N', 
                    help='input batch size for training (default: 64)') 
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N', 
                    help='input batch size for testing (default: 1000)') 
parser.add_argument('--epochs', type=int, default=10, metavar='N', 
                    help='number of epochs to train (default: 10)') 
parser.add_argument('--lr', type=float, default=0.01, metavar='LR', 
                    help='learning rate (default: 0.01)') 
parser.add_argument('--momentum', type=float, default=0.5, metavar='M', 
                    help='SGD momentum (default: 0.5)') 
parser.add_argument('--no-cuda', action='store_true', default=False, 
                    help='disables CUDA training') 
parser.add_argument('--seed', type=int, default=1, metavar='S', 
                    help='random seed (default: 1)') 
parser.add_argument('--log-interval', type=int, default=10, metavar='N', 
                    help='how many batches to wait before logging training status') 
parser.add_argument('--save-model', action='store_true', default=False, 
                    help='For Saving the current Model')

parser.print_help()

args_str = '--batch-size 10' 
FLAGS, _ = parser.parse_known_args(args=args_str.split())
print(FLAGS)

# ## easydict

import easydict 
args = easydict.EasyDict({ 
    "batchsize": 100, 
    "epoch": 20, 
    "gpu": 0, 
    "out": "result", 
    "resume": False, 
    "unit": 1000 
})


