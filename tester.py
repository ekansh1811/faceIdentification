import pandas as pd
import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from PIL import Image, ImageOps
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)


class ConvNeuralNet(nn.Module):
  def __init__(self, num_classes):
    super(ConvNeuralNet, self).__init__()
    self.conv_l1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=2)
    self.max_pool1 = nn.MaxPool2d(kernel_size=4,stride=3)
    
    self.conv_l2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=2)
    self.max_pool2 = nn.MaxPool2d(kernel_size=4, stride=3)
    self.fc1 = nn.Linear(512,256)
    self.relu1 = nn.ReLU()
    self.fc2 = nn.Linear(256,num_classes)

  def forward(self, x):
    out = self.conv_l1(x)
    out = self.max_pool1(out)
    out = self.conv_l2(out)
    
    out = self.max_pool2(out)

    out = out.reshape(out.size(0), -1)

    out = self.fc1(out)
    out = self.relu1(out)
    out = self.fc2(out)
    return out
  
device = torch.device('cpu')

num_classes=4

model = ConvNeuralNet(num_classes).to(device)
model.load_state_dict(torch.load("model.pth"))

while True:
    with torch.no_grad():
        _, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.imshow('img', img)
            images = gray[y:y+h,x:x+w]
            images = cv2.resize(images,(48,48))
            sc = StandardScaler()
            images = sc.fit_transform(images)
            images = torch.tensor(images,dtype=torch.float32)
            images = images.resize_(1,1,48,48)
            images = images.to(device)
            outputs = model(images)
            _,predicted = torch.max(outputs.data,1)
            print(predicted)
        k = cv2.waitKey(30) & 0xff
        if k==27:
            break



cap.release()