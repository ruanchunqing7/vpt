
import torch
import torchvision
import pickle
from torch import nn

# vgg16 = torchvision.models.vgg16(pretrained=False)
# torch.save(vgg16.state_dict(), "pre-trained_weights/vgg16_model.pth")
# vgg16.load_state_dict(torch.load("pre-trained_weights/vgg16_model.pth"))

# with open("shot_16-seed_1.pkl", "rb") as file:
file = open("shot_16-seed_1.pkl", "rb")
data = pickle.load(file)
print(data)
train, val = data["train"], data["val"]


 # import cPickle as pickle
 #    f = open('path')
 #    info = pickle.load(f)
 #    print info