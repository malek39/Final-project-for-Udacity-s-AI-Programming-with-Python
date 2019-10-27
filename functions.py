import matplotlib.pyplot as plt
import torch 
import numpy as np 
from torch import nn 
from torch import optim 
import torch.nn.functional as F
from torchvision import datasets , transforms , models 
from collections import OrderedDict 
from PIL import Image 

import argparse


def arg_parser_train():
    # Define parser
    parser = argparse.ArgumentParser(description="Neural Network Settings")
    
    parser.add_argument('data_dir', nargs='*', action="store", default="ImageClassifier/flowers/")
    parser.add_argument('--gpu', type=bool, default=False, help='Use GPU or not')
    parser.add_argument('--arch', type=str, default='vgg16', help='architecture [available: densenet, vgg]')
    parser.add_argument('--dropout', action = 'store', type=int, default = 0.05,
                    help = 'Enter dropout for training the model, default is 0.05.')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--hidden_units', type=int, default=4096, help='hidden units for fc layer')
    parser.add_argument('--epochs', type=int, default=3, help='number of epochs')
    parser.add_argument('--save_dir' , type=str, default='checkpoint.pth', help='path of your saved model')
    # Parse args
    args = parser.parse_args()
    return args

# TODO: Define your transforms for the training, validation, and testing sets
def train_transformer_load(train_dir , valid_dir , test_dir):
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
    
    # TODO: Load the datasets with ImageFolder
    image_datasets = dict()
    image_datasets['train'] = datasets.ImageFolder(train_dir, transform=train_transforms)
    image_datasets['valid'] = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    image_datasets['test'] = datasets.ImageFolder(test_dir, transform=test_transforms)
    
    # TODO: Using the image datasets and the trainforms, define the dataloaders

    dataloaders = dict()
    dataloaders['train']=torch.utils.data.DataLoader(image_datasets['train'] , batch_size=64 , shuffle=True)
    dataloaders['valid']=torch.utils.data.DataLoader(image_datasets['valid'] , batch_size=32 , shuffle=True)
    dataloaders['test']=torch.utils.data.DataLoader(image_datasets['test'] , batch_size=16 , shuffle=True)
    
    return image_datasets , dataloaders


# Load a pre_trained_network
arch = {"vgg16":25088,
        "densenet121":1024
        }
def pre_trained_network(structure='vgg16',dropout=0.5, hidden_layer1 = 4096,lr = 0.001):
    
    #Selection of the archt
    if structure == 'vgg16':
        model = models.vgg16(pretrained=True)
        
    elif structure == 'densenet121':
        model = models.densenet121(pretrained=True)
    
    else:
        print("Please try for vgg16 or densenet121")
        
    
        
    for param in model.parameters():
        param.requires_grad = False
        
    #Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(arch[structure],hidden_layer1)),
                          ('relu1', nn.ReLU()),
                          ('d_out1',nn.Dropout(dropout)),
                          ('fc2', nn.Linear(hidden_layer1, 1024)),
                          ('relu2', nn.ReLU()),
                          ('d_out2',nn.Dropout(dropout)),
                          ('fc3', nn.Linear(1024, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
        
        
    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr )
    model.cuda()
        
    return model, optimizer ,criterion

def learning(model, trainloader, validloader, epochs, print_every, criterion, optimizer, device='cpu'):
    print_every = print_every
    steps = 0

    # change to cuda
    model.to('cuda')

    for e in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1

            inputs, labels = inputs.to('cuda'), labels.to('cuda')

            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                v_lost = 0
                v_accuracy=0
                for ii, (inputs2,labels2) in enumerate(validloader):
                    optimizer.zero_grad()
                    inputs2, labels2 = inputs2.to('cuda') , labels2.to('cuda')
                    model.to('cuda')
                    with torch.no_grad():    
                        outputs = model.forward(inputs2)
                        v_lost = criterion(outputs,labels2)
                        ps = torch.exp(outputs).data
                        equality = (labels2.data == ps.max(1)[1])
                        v_accuracy += equality.type_as(torch.FloatTensor()).mean()
                    
                v_lost = v_lost / len(validloader)
                v_accuracy = v_accuracy /len(validloader)
            
                print("Epoch: {}/{}... ".format(e+1, epochs),
                  "Loss: {:.4f}".format(running_loss/print_every),
                  "Validation Lost {:.4f}".format(v_lost),
                   "Accuracy: {:.4f}".format(v_accuracy))
                running_loss = 0

  
def check_accuracy_on_test(testloader , model):  
    correct = 0
    total = 0
    model.to('cuda')
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

def save_checkpoint(model, Save_Dir, Train_data, structure, hidden_layer , learning_rate , dropout ):   
    # Save model at checkpoint
    if type(Save_Dir) == type(None):
        print("Model checkpoint directory not specified, model will not be saved.")
    else:
        
        # Create `class_to_idx` attribute in model
        model.class_to_idx = Train_data.class_to_idx
            
        # Create checkpoint dictionary and Save checkpoint
        model.cpu
        torch.save({'arch' : structure,
            'hidden_layer1': hidden_layer,
            'learning_rate': learning_rate,
            'dropout': dropout,       
            'state_dict':model.state_dict(),
            'class_to_idx':model.class_to_idx},
            'checkpoint.pth')
            

#####################################################################################################
#####################################################################################################     
#####################################################################################################       
def arg_parser_test():
    # Define parser
    parser = argparse.ArgumentParser(description="Neural Network Settings")
    
    parser.add_argument('image_path', action='store',nargs='?' , type = str 
                    ,default = 'ImageClassifier/flowers/test/102/image_08004.jpg',
                    help='Enter path to image.' )

    parser.add_argument('checkpoint', action='store',nargs='?' , type = str , 
                    default = 'checkpoint.pth' , help='path of your saved model')

    parser.add_argument('--top_k', action='store',
                    dest='top_k', type=int, default = 5,
                    help='Enter number of top most likely classes to view, default is 3.')

    parser.add_argument('--cat_to_name', action='store',
                    dest='cat_name_dir', default = 'ImageClassifier/cat_to_name.json',
                    help='Enter path to image.')

    
    # Parse args
    args = parser.parse_args()
    return args



def load_model(path):
    checkpoint = torch.load(path)
    arch = checkpoint['arch']
    hidden_layer1 = checkpoint['hidden_layer1']
    learning_rate = checkpoint['learning_rate']
    dropout = checkpoint['dropout']
    model,_,_ = pre_trained_network(arch , dropout,hidden_layer1 ,learning_rate)
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    test_image = Image.open(image)
    prepoceess_img = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),  
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = prepoceess_img(test_image)
    return img_tensor
    
def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax


def predict(image_path , model, topk=5 , device = 'gpu'):   
    if device == 'gpu':
        model.to('cuda')
    img_torch = process_image(image_path)
    img_torch = img_torch.unsqueeze_(0)
    img_torch = img_torch.float()
    
    if device == 'gpu':
        with torch.no_grad():
            output = model.forward(img_torch.cuda())
    
    else:
        with torch.no_grad():
            output = model.forward(img_torch)
        
    probability = F.softmax(output.data,dim=1)
    
    return probability.topk(topk)

def check_sanity(path):
    plt.rcParams["figure.figsize"] = (10,5)
    plt.subplot(211)
    
    index = 1
    #path = test_dir + path


    probabilities = predict(path, model)
    image = process_image(path)
    probabilities = probabilities
    

    axs = imshow(image, ax = plt)
    axs.axis('off')
    axs.title(cat_to_name[str(index)])
    axs.show()
    
    
    a = np.array(probabilities[0][0])
    b = [cat_to_name[str(index+1)] for index in np.array(probabilities[1][0])]
    
    
    N=float(len(b))
    fig,ax = plt.subplots(figsize=(8,3))
    width = 0.8
    tickLocations = np.arange(N)
    ax.bar(tickLocations, a, width, linewidth=4.0, align = 'center')
    ax.set_xticks(ticks = tickLocations)
    ax.set_xticklabels(b)
    ax.set_xlim(min(tickLocations)-0.6,max(tickLocations)+0.6)
    ax.set_yticks([0.2,0.4,0.6,0.8,1,1.2])
    ax.set_ylim((0,1))
    ax.yaxis.grid(True)
    
    plt.show()

