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
import train



#define argparse fnc to get path to image, model checkpoint, mapping of categories to names
def arg_parse():
    parser = argparse.ArgumentParser(description = "Use Neural Network for prediction")
    parser.add_argument("image_path", help = "specify path to image ")
    parser.add_argument("checkpoint", help ="specify a Neural Network model checkpoint to use for prediction")
    parser.add_argument("--top_k", help ="number of most probable names", type = int, default = 1)
    parser.add_argument("--category_names", help = "specify path to file mapping category to real name",
                        default = "cat_to_name.json")
    parser.add_argument("--gpu", help = "choose between using gpu or cpu", action = "store_true")
    args = parser.parse_args()
    
    return args



def load_model(checkpoint):
    #load checkpoint
    checkpoint = torch.load(checkpoint)
    
    #download pretrained base model architecture
    if checkpoint['model_name'] == 'vgg16':
        model = models.vgg16(pretrained = True)
    else:
        model = models.vgg13(pretrained = True)
     
    #add classifier
    model.classifier = checkpoint['classifier']
    
    #add trained model attributes
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model


# Function to preprocess image used for prediction
def process_image(path):
    img = Image.open(path)
    
    #transforming image so it fits model parameters
    img_transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                                     transforms.Normalize([.485,.456,.406], [.229,.224,.225])])
    return img_transform(img)
    

#Function returns prediction of top probabilities and top classes
def predict(path, model , top_probs, cat_mapping, device):
    #use GPU if specified and if available
    if device:
        if torch.cuda.is_available():
            device_use = 'cuda'
    else:
        device_use = 'cpu'
    
    #preprocess image used for prediction
    image = process_image(path)
    
    image.unsqueeze_(0)
    image = image.to(device_use)
    
    #forward pass throught the model
    with torch.no_grad():
        model.to(device_use)
        model.eval()
        prediction = model(image)
        probabilities = torch.exp(prediction)
        top_prob,top_class =  probabilities.topk(top_probs,dim=1)
        model.train()
        
        categories = []
        cat_name = []
        #inverting class_to_idx dict
        for i, idx in enumerate(top_class.cpu().view(len(top_class[0]),).numpy()):
            for k,v in model.class_to_idx.items():
                if idx == v:
                    categories.append(k)
        
        
        #mapping index to flower names
        with open(cat_mapping, 'r') as f:
            cat_to_name = json.load(f)
        
        for i in categories:
            for k,v in cat_to_name.items():
                if i==k:
                    cat_name.append(v)
        
    
    return top_prob.cpu().view(len(top_prob[0]),).numpy().tolist(), cat_name
    



def main():
    
    #parse through cmd line for path to image, model checkpoint, mapping of categories to names 
    args = arg_parse()

    #initialize variables with values received from arg_parse()
    path = args.image_path
    checkpoint_model = args.checkpoint
    top_probs = args.top_k
    cat_mapping = args.category_names
    device = args.gpu
                  
    #construct model with saved checkpoint
    model = load_model(checkpoint_model)
    
    #Predict top probabilities, classes
    prob, class_name = predict(path, model , top_probs, cat_mapping, device)
    
    #Display the top classes and probabilities
    print('The probabilities are: {}'.format(prob))
    print('The flowers corresponding to the probabilities are: {}'.format(class_name))
    
    
    
if __name__ == '__main__':
    main()