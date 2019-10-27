import matplotlib.pyplot as plt
import torch 
import numpy as np 
from torch import nn 
from torch import optim 
import torch.nn.functional as F
from torchvision import datasets , transforms , models 
from collections import OrderedDict 
from PIL import Image 


from functions import arg_parser_train, train_transformer_load, pre_trained_network,learning, check_accuracy_on_test, save_checkpoint, load_model ,predict

def main():    
    
    args = arg_parser_train()
    #Loading the data
    data_dir = args.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    image_datasets , dataloaders = train_transformer_load(train_dir , valid_dir , test_dir)
    
    # Label mapping
    import json
    with open('ImageClassifier/cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    
    # Load a pre_trained_network
    model,optimizer,criterion = pre_trained_network(args.arch, args.dropout, args.hidden_units, args.lr  )

    learning(model, dataloaders['train'],dataloaders['valid'],args.epochs, 40, criterion, optimizer, args.gpu)
    check_accuracy_on_test(dataloaders['test'],model)
    
    # TODO: Save the checkpoint
    save_checkpoint(model, args.save_dir, image_datasets['train'], args.arch, args.hidden_units, args.lr, args.dropout)

    
# =============================================================================
# Run Program
# =============================================================================
if __name__ == '__main__': main()