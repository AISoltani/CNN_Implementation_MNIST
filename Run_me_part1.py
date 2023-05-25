# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 10:49:17 2020

@author: AISoltani
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

n_epochs = 4
batch_size_train = 60
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)
train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('/files/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.Resize((32,32)),
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('/files/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.Resize((32,32)),
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_test, shuffle=True)

examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1,64,3,1,1)
        self.conv2 = nn.Conv2d(64,128,3,1,1)
        self.conv3 = nn.Conv2d(128,256,3,1,1)
        self.conv4 = nn.Conv2d(256,256,3,1,1)
        self.conv5 = nn.Conv2d(256,512,3,1,1)
        self.conv6 = nn.Conv2d(512,512,3,1,1)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)
        self.fc1 = nn.Linear(512,4096)
        self.fc2 = nn.Linear(4096,4096)
        self.fc3 = nn.Linear(4096,10)
        self.Drop = nn.Dropout2d(0.5)
    def forward(self,x): 
        ##1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x,(2,2))
        ##2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x,(2,2))
        ##3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        ##4
        x = self.conv4(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.max_pool2d(x,(2,2))
        ##5
        x = self.conv5(x)
        x = self.bn4(x)
        x = F.relu(x)
        ##6
        x = self.conv6(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = F.max_pool2d(x,(2,2))
        ##7
        x = self.conv6(x)
        x = self.bn4(x)
        x = F.relu(x)
        ##8
        x = self.conv6(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = F.max_pool2d(x,(2,2))
        ##9
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc1(x)
        x = F.relu(x)
        x = self.Drop(x)
        #10
        x = self.fc2(x)
        x = F.relu(x)
        x = self.Drop(x)
        ##11
        x = self.fc3(x)

        return F.log_softmax(x)
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
network = Net()
optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
Accuracy_train = []
Accuracy_test = []
train_losses = []
train_losses_epoch = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]
criterion = nn.CrossEntropyLoss()

def train(epoch):
  network.train()
  correct_train = 0
  for batch_idx, (data, target) in enumerate(train_loader):
    optimizer.zero_grad()
    output = network(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    prediction = output.data.max(1, keepdim=True)[1]
    correct_train += prediction.eq(target.data.view_as(prediction)).sum()
    # if batch_idx % log_interval == 0:
    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: [{}/{} ({:.0f}%)]  '.format(
      epoch, (batch_idx+1) * len(data), len(train_loader.dataset),
      100. * batch_idx / len(train_loader), loss.item(),np.array(correct_train),(batch_idx+1) * len(data),
      100. * np.array(correct_train)/((batch_idx+1) * len(data))))
    train_losses.append(loss.item())
    train_counter.append((batch_idx*60) + ((epoch-1)*len(train_loader.dataset)))
  train_losses_epoch.append(loss.item())
  Accuracy_train.append(100. * np.array(correct_train)/((batch_idx+1) * len(data)))
  print('Train Epoch: {} \tAvg. Loss: {:.6f}\tAccuracy: [{}/{} ({:.0f}%)]  '.format(
        epoch, loss.item(),np.array(correct_train),(batch_idx+1) * len(data),
        100. * np.array(correct_train)/((batch_idx+1) * len(data))))

     
def test():
  network.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      output = network(data)
      test_loss += criterion(output, target)#, size_average=False).item()
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()
  test_loss /= len(test_loader.dataset)
  test_losses.append(test_loss)
  Accuracy_test.append(100. * correct / len(test_loader.dataset))
  print('\nTest set: Avg. loss: {:.6f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))
  
for epoch in range(1, n_epochs + 1):
  train(epoch)
  test()

fig = plt.figure()
plt.scatter(np.array(range(1,n_epochs+1)), np.array(train_losses_epoch[0:n_epochs]), color='blue')
plt.xlabel('number of epochs')
plt.ylabel('Train Cross-entropy Loss')

fig = plt.figure()
plt.scatter(np.array(range(1,n_epochs+1)), np.array(test_losses[0:n_epochs]).T, color='blue')
plt.xlabel('number of epochs')
plt.ylabel('Test Cross-entropy Loss')

fig = plt.figure()
plt.scatter(np.array(range(1,n_epochs+1)), np.array(Accuracy_train[0:n_epochs]), color='blue')
plt.xlabel('number of epochs')
plt.ylabel('Train Accuracy %')

fig = plt.figure()
plt.scatter(np.array(range(1,n_epochs+1)), np.array(Accuracy_test[0:n_epochs]), color='blue')
plt.xlabel('number of epochs')
plt.ylabel('Test Accuracy %')
plt.show()

print('Train_L',train_losses_epoch[0:n_epochs])
print('Test_l',np.array(test_losses[0:n_epochs]).T)
print('Train_A',Accuracy_train[0:n_epochs])
print('test_A',np.array(Accuracy_test[0:n_epochs]))
