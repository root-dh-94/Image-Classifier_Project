import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
from torch import optim
from torchvision import datasets, transforms, models
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt 
import argparse
import json





#define argparse fnc to get hyperparameters, model architecture
def arg_parse():
    parser = argparse.ArgumentParser(description = "Customize Neural Network")
    parser.add_argument("data_directory", help = "specify path to data used to train Neural Network")
    parser.add_argument("--save_dir", help ="specify directory to save checkpoints of trained Neaural Network")
    parser.add_argument("--arch", help ="choose model architecture from either vgg16 or vgg13", choices = ["vgg16","vgg13"],
                        default = "vgg16")
    parser.add_argument("--learning_rate", help = "specify learning rate for gradient descent", dest = "lr", default = .001, 
                        type = float)
    parser.add_argument("--hidden_units", help = "specify size of hidden layer, number between 1024 and 25088", default = 4096, 
                        type = int, choices = range(1024,25088))
    parser.add_argument("--epochs", help = "specify number of epochs the Neural Network is trained with", default = 13, 
                        type = int)
    parser.add_argument("--gpu", help = "choose between using gpu or cpu", action = "store_true")
    args = parser.parse_args()
    
    return args





#define function for loading and processing data for training/testing/validation
def preprocess_data(data_dir):
    #specifying the path for each data set
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    #Specifying transforms for the training, validation, and testing sets
    data_transforms_train = transforms.Compose([transforms.RandomRotation(30), transforms.RandomResizedCrop(224),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(), transforms.Normalize([.485,.456,.406], [.229,.224,.225])])
    
    data_transforms_val = transforms.Compose([transforms.Resize(255), transforms.CenterCrop(224),
                                              transforms.ToTensor(), transforms.Normalize([.485,.456,.406], [.229,.224,.225])])
    
    data_transforms_test = transforms.Compose([transforms.Resize(255), transforms.CenterCrop(224),
                                               transforms.ToTensor(), transforms.Normalize([.485,.456,.406], [.229,.224,.225])])
    
    #Loading the datasets with ImageFolder
    image_datasets_train = datasets.ImageFolder(train_dir,transform = data_transforms_train)
    image_datasets_val = datasets.ImageFolder(valid_dir,transform = data_transforms_val)
    image_datasets_test = datasets.ImageFolder(test_dir,transform = data_transforms_test)

    #Finalizing training, validation and testing sets
    dataloaders_train = torch.utils.data.DataLoader(image_datasets_train, batch_size = 32, shuffle = True)
    dataloaders_val = torch.utils.data.DataLoader(image_datasets_val, batch_size = 32, shuffle = True)
    dataloaders_test = torch.utils.data.DataLoader(image_datasets_test, batch_size = 32, shuffle = True)
    
    return dataloaders_train, dataloaders_val, dataloaders_test, image_datasets_train




#TRAINING AND VALIDATING MODEL
def train_val_model(dataloaders_train, dataloaders_val, image_datasets_train,
                    base_model, learning_rate, hidden_layer, ep, device, checkpoint_path):
    #use GPU if specified and if available
    if device:
        if torch.cuda.is_available():
            device_use = 'cuda'
    else:
        device_use = 'cpu'
    
    #download pretrained base model architecture
    if base_model == 'vgg16':
        model = models.vgg16(pretrained = True)
    else:
        model = models.vgg13(pretrained = True)
        
    #freeze pretrained model paramters
    for param in model.parameters():
        param.requires_grad = False
    
    #Create new feedforward nerwork and update classifier
    classifier = nn.Sequential(nn.Linear(25088, hidden_layer), nn.ReLU(), nn.Dropout(p = 0.4),
                               nn.Linear(hidden_layer, 1024), nn.ReLU(), nn.Dropout(p = 0.4),
                               nn.Linear(1024, 102), nn.LogSoftmax(dim = 1))
    
    #add new feedforward network to base model to create final model architecure
    model.classifier = classifier
    
    # initialize variables for gradient descent
    optimizer = optim.Adam(model.classifier.parameters(), learning_rate)
    criterion = nn.NLLLoss()
    
    #training the new feedforward classifier
    model.to(device_use)
    for i in range(ep):
        train_loss = 0
    
        for images, labels in dataloaders_train:
        
            images, labels = images.to(device_use), labels.to(device_use)
            optimizer.zero_grad()
            prediction = model(images)
            loss = criterion(prediction, labels)
            loss.backward()
            optimizer.step()
        
            train_loss += loss.item()
            
        #implementing validation run
        with torch.no_grad():
            model.eval()
            accuracy = 0
            val_loss = 0
            
            for images, labels in dataloaders_val:
                images, labels = images.to(device_use), labels.to(device_use)
                prediction = model(images)
                loss = criterion(prediction, labels)
                val_loss += loss.item()

                ps = torch.exp(prediction)
                top_prob, top_class = ps.topk(1, dim = 1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            
            model.train()
                
            print('Training loss:', train_loss/len(dataloaders_train) )
            print('Validation loss:', val_loss/len(dataloaders_val) )
            print('Accuracy:', accuracy/len(dataloaders_val) )
    
    #save checkpoint if path specified
    if checkpoint_path:
        model.cpu()
        
        #maps label indeces to class
        model.class_to_idx = image_datasets_train.class_to_idx
        
        #save model attributes
        checkpoint = {'classifier': model.classifier,
                      'class_to_idx': model.class_to_idx,
                      'state_dict': model.state_dict(),
                      'model_name': base_model
                     }
        
        #save model
        torch.save(checkpoint, checkpoint_path)
        
        
        

    


def main():
    #parse through cmd line for path to data set, hyperparameters, model architecure 
    args = arg_parse()

    #initialize variables with values received from arg_parse()
    data_dir = args.data_directory
    
    if args.save_dir:
        checkpoint_path = args.save_dir
    else:
        checkpoint_path = False
   
    base_model = args.arch
    learning_rate = args.lr
    hidden_layer = args.hidden_units
    ep = args.epochs
    device = args.gpu
    
    #getting datasets
    dataloaders_train, dataloaders_val, dataloaders_test, image_datasets_train = preprocess_data(data_dir)
    
    # Start training and validating the model
    train_val_model(dataloaders_train, dataloaders_val, image_datasets_train, base_model, 
                    learning_rate, hidden_layer, ep, device, checkpoint_path)
    
    
    
    
if __name__ == '__main__':
    main()
    
    