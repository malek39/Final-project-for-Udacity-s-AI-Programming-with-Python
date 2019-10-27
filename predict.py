import matplotlib.pyplot as plt
import torch 
import numpy as np 
from torch import nn 
from torch import optim 
import torch.nn.functional as F
from torchvision import datasets , transforms , models 
from collections import OrderedDict 
from PIL import Image 
from functions import load_model , arg_parser_test , imshow , process_image , predict ,check_sanity
import json

def main():
    #Args
    args1 = arg_parser_test()
    #Load the model
    model=load_model(args1.checkpoint)  
    #label 
    with open(args1.cat_name_dir,'r') as json_file:
        cat_to_name = json.load(json_file)
    #Prediction
    probabilities = predict(args1.image_path, model, args1.top_k)
    labels = [cat_to_name[str(index + 1)] for index in np.array(probabilities[1][0])]
    probability = np.array(probabilities[0][0])
    i=0
    while i < args1.top_k:
        print("{} with a probability of {}".format(labels[i], probability[i]))
        i += 1
    print("Predictiion is done !")

# =============================================================================
# Run Program
# =============================================================================
if __name__ == '__main__': main()