"""

- By JIAYI CHEN
- (1) The code, including how to build, train and test a cnn, was written with the help of the code from cnn Labs by Professor Tom Gedeon, ANU
- (2) The main technique of the code: Autoencoder was inspired by the paper: Image Compression using Shared Weights and Bidirectional Networks
- (3) Thanks to Xuanyu Chen, I have become clear about how to implement a cnn autoencoder, that is to build a cnn, where convolutional layers and full connection layers are opposite,
   the dimension of convolutional layers should increase then decrease, and dimension of full connection layers should decrease then increase
- (4) Thanks to Ao Feng, I have become clear about how to use view function to flatten or unflatten the cnn in the AutoEncoder, which is swapping in 4D and 2D.
   Moreover, the size of the input data would affect the parameter that view function takes, so it's better to include it in the definition of AutoEncoder.

"""

import os
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as op
import torchvision.datasets as dt
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as trans
import matplotlib.pyplot as plt

learning_rate = 0.01
batch_size_ae = 1000
batch_size = 100
num_epochs_ae = 10
num_epochs = 100

# load dataset
train_set = dt.MNIST('./data', train=True, transform=trans.Compose([trans.ToTensor(), trans.Normalize((0,), (1.0,))]))
test_set = dt.MNIST('./data', train=False, transform=trans.Compose([trans.ToTensor(), trans.Normalize((0,), (1.0,))]))

# set train loader
train_loader_c = torch.utils.data.DataLoader(dataset=train_set,batch_size=batch_size_ae,shuffle=True)

# define AutoEncoder
class AutoEncoder(nn.Module): #(2)(3)
    def __init__(self):
        super().__init__()
        # Encoder
        self.cnn_1 = nn.Conv2d(1, 16, 5)
        self.cnn_2 = nn.Conv2d(16, 32, 5)
        self.linear_1 = nn.Linear(4 * 4 * 32, 60)
        self.linear_2 = nn.Linear(60, 16)     # 28*28 -> 16
        # Decoder                             # can be encoded to different size by changing these two 16 to the preferred size
        self.linear_3 = nn.Linear(16, 200)    # 16 -> 28*28
        self.linear_4 = nn.Linear(200, 28*28)

    def forward(self, data):
        # Encoder
        encoded = self.cnn_1(data)
        encoded = F.max_pool2d(encoded, 2)
        encoded = F.rrelu(encoded)
        encoded = self.cnn_2(encoded)
        encoded = F.max_pool2d(encoded, 2)
        encoded = F.rrelu(encoded)
        encoded = encoded.view([data.size(0), -1]) #(4)
        encoded = self.linear_1(encoded)
        encoded = F.rrelu(encoded)
        encoded = self.linear_2(encoded)
        # Decoder
        decoded = self.linear_3(encoded)
        decoded = F.rrelu(decoded)
        decoded = self.linear_4(decoded)
        decoded = F.rrelu(decoded)
        decoded = decoded.view([encoded.size(0), 1, 28, 28]) #(4)
        return decoded, encoded

# Training
autoencoder = AutoEncoder()
if torch.cuda.is_available():
    autoencoder = autoencoder.cuda()
loss_fn = nn.MSELoss()
optimizer = op.Adam(autoencoder.parameters(), lr=learning_rate)

for epoch in range(num_epochs_ae):
    for batch, (data, target) in enumerate(train_loader_c):
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        data = Variable(data)
        decoded, encoded = autoencoder(data)
        target = Variable(target)
        optimizer.zero_grad()
        loss = loss_fn(decoded, data)
        loss.backward()
        optimizer.step()
        #if (batch+1) == len(train_loader_c):
        print ('autoencoder training - ','epoch:',epoch, 'batch:', batch+1, 'loss:', loss.data[0])

"""
#show input image
image = data.cpu().data
image = image / 2 + 0.5
image = image.clamp(0,1)
image = image.view(image.size(0),1,28,28)
img = trans.ToPILImage()(image[0])
plt.imshow(img)
plt.show()

#show processed image image
image = decoded.cpu().data
image = image / 2 + 0.5
image = image.clamp(0,1)
image = image.view(image.size(0),1,28,28)
img = trans.ToPILImage()(image[0])
plt.imshow(img)
plt.show()
"""

# set train loader and test loader
dataset_C = torch.utils.data.TensorDataset(data_tensor=decoded.data, target_tensor=target.data)
train_loader = torch.utils.data.DataLoader(dataset_C, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_set,batch_size=batch_size,shuffle=False)

# define classification network
class Net(nn.Module): #(1)
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 50, 5)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# training
net = Net()
if torch.cuda.is_available():
    net = net.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = op.SGD(net.parameters(), lr=learning_rate, momentum=0.5)

for epoch in range(num_epochs):
    for batch, (data, target) in enumerate(train_loader):
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = net(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if (batch+1) == len(train_loader):
            print ('classification cnn training - ','epoch:',epoch, 'loss:', loss.data[0])

# testing
correct = total = 0
for batch, (data, target) in enumerate(test_loader):
    if torch.cuda.is_available():
        data, target = data.cuda(), target.cuda()
    data, target = Variable(data), Variable(target)
    output = net(data)
    loss = criterion(output, target)
    _, predicted = torch.max(output.data, 1)
    total += data.data.size()[0]
    correct += (predicted == target.data).sum()
    if (batch+1) == len(test_loader):
        print ('testing - ','Loss:', loss.data[0],'accuracy:', correct * 1.0 * 100/ total,'%' )
