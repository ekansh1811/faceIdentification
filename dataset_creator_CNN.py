import pandas as pd
import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from PIL import Image, ImageOps

class FeatureDataset(Dataset):
    def __init__(self,file_name):
        file_out = pd.read_csv(file_name)

        x = file_out.iloc[0:441,1].values
        y = file_out.iloc[0:441,0].values
        print(x)
        z = np.empty((440,48,48),dtype = float)
        k = 0
        for i in x:
            
            print(i)
            j = "data/faces/"+i

            print(j)
            img = Image.open(j)
            img = ImageOps.grayscale(img)
            z[k] = np.array(img)
            k+=1
        print(z)
        x = z
        sc = StandardScaler()
        x.shape = (440,2304)
        x_train = sc.fit_transform(x)
        y_train = y

        self.X_train = torch.tensor(x_train,dtype=torch.float32)
        self.y_train = torch.tensor(y_train)

    def __len__(self):
        return len(self.y_train)
    def __getitem__(self,idx):
        return self.X_train[idx],self.y_train[idx]
    


feature_set = FeatureDataset("data/data.csv")


train_loader = DataLoader(feature_set,batch_size=110,shuffle=True)

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
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01, weight_decay = 0.005, momentum = 0.9)
total_step = len(feature_set)

epochs = 4

for e in range(epochs):
    for images, labels in train_loader:
        images = images.reshape(110,1,48,48).to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs,labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('Epoch [{}], loss:{:.4f}'.format(e,loss.item()))


torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")