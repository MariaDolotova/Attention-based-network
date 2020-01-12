import _pickle as cPickle
from .resnet18_final import resnet18
import torch


#load weights to modified resnet18 model

model=resnet18()

input = open('resnet18_weights.pkl', 'rb')
state_dict = cPickle.load(input)
input.close() 


pretrained_dict = state_dict
model_dict = model.state_dict()
# filter out unnecessary keys
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
#overwrite entries in the existing state dict
model_dict.update(pretrained_dict) 

#load the new state dict
model.load_state_dict(model_dict)
torch.save(model, 'pretrained_resnet18.pth')
    