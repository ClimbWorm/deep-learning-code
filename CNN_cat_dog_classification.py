import pandas as pd
import numpy as np
import sys
import sklearn
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from torchvision import transforms, datasets
from tqdm import tqdm
import matplotlib.pyplot as plt
# from matplotlib.pyplot import imshow
from helper.visualize import imshow, show_example

"""
The pipeline of this churn prediction classification problem based on Convolutional Neural Network:
data preprocessing(data augmentation)
train_test_split
feature scaling
model building
model training
model evaluation
prediction
"""
# https://www.udemy.com/course/deeplearning/learn/lecture/6905308#questions/5254500


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# create train_trasform and test_transform for data augmentation
# the purpose of data augmentation: prevent overfitting
train_transform = Compose([
    transforms.RandomHorizontalFlip(p=0.3),
    # transforms.RandomCrop(32, padding=4), # this crop will destroy the original feature a lot, do not use it for now
    transforms.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75)),
    transforms.ToTensor(),  # Convert a PIL Image or numpy.ndarray to tensor.
    transforms.Resize(size=(64, 64)),  # if time allowed, we can choose larger size
    transforms.Normalize([0, 0, 0], [1, 1, 1])  # since normalize can only apply on tensor, so put it in the end
])
test_transform = Compose([
    # cannot use flip & affine, it's similar to fit_transform on training set and only transform on testing set
    transforms.ToTensor(),
    transforms.Resize(size=(64, 64)),
    transforms.Normalize([0, 0, 0], [1, 1, 1])
])

dataset_train = datasets.ImageFolder(root='./dataset/training_set/', transform=train_transform)
train_loader = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=32,
                                           shuffle=True)  # it's not good to read all the data to RAM at once, use dataloader instead

dataset_test = datasets.ImageFolder(root='./dataset/test_set/', transform=test_transform)
test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=32, shuffle=True)


# build CNN model
class ConvNet(torch.nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv = torch.nn.Sequential(
            # floor((shape + 2p - f)/2+1)
            torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(2, 2), stride=2),  # [32,64,32,32]
            torch.nn.ReLU(),  # [32,64,32,32]
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2),  # [32,64,16,16]

            torch.nn.Conv2d(64, 16, kernel_size=(3, 3), padding=1),  # [32,16,16,16]
            torch.nn.ReLU(),  # [32,16,16,16]
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2),  # [32,16,8,8]
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(16 * 8 * 8, 360),
            torch.nn.ReLU(),
            torch.nn.Linear(360, 20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, 1),  # number of output neural = 1 because it is a binary classification
            torch.nn.Sigmoid(),  # Note in multiclass classification, softmax already included in the cross entropy loss
        )

    def forward(self, x):
        out = self.conv(x)
        out = out.view(-1, 16 * 8 * 8)  # flatten, the first argument here is batch size
        out = self.fc(out)
        return out


# model training
# define how to train the model
def train(net, tr_loader, te_loader, n_epochs=100, learning_rate=0.01):
    """
    :param net: the architecture we built before, e.g.CNN(torch.nn.Module)
    :param tr_loader: train_loader we created before using dataloader
    :param te_loader: test_loader we created before using dataloader
    :param n_epochs: training epochs
    :param learning_rate: learning rate
    :return: no
    """
    # list_of_losses = []
    # loss = torch.nn.CrossEntropyLoss() # this is for multi-class classification
    # print(net)
    loss = torch.nn.BCELoss()
    print(net.parameters())
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)  # tune the learning rate, optimize the result
    iter = 0
    for _ in tqdm(range(n_epochs)):
        for i, (images, labels) in enumerate(tr_loader):
            images = images.to(device).float()
            labels = labels.to(device).float()
            # Load images
            # images = images.requires_grad_()

            # Forward pass to get output/logits
            y_pred = net(images)  # forward

            # Calculate Loss: softmax --> cross entropy loss
            l = loss(y_pred.squeeze(1), labels)
            # list_of_losses.append(l.item())

            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()
            # Getting gradients w.r.t. parameters
            l.backward()
            # Updating parameters
            optimizer.step()

            iter += 1
            # net.eval()
            if iter % 10 == 0:
                # Calculate Accuracy
                correct = 0
                total = 0
                # Iterate through test dataset
                for image, label in te_loader:
                    # Load images
                    image = image.requires_grad_()

                    # Forward pass only to get logits/output
                    outputs = net(image)

                    # Get predictions from the maximum value
                    # _, predicted = torch.max(outputs.data, 1) # this is for multi-class
                    predicted = outputs.squeeze(1) > 0.5

                    # Total number of labels
                    total += label.size(0)

                    # Total correct predictions
                    correct += (predicted == label).sum()

                accuracy = 100 * correct / total

                # Print Loss
                print('Iteration: {}. Loss: {}. Accuracy: {}'.format(iter, l.item(), accuracy))

    return None  # list_of_losses


if __name__ == "__main__":
    # Run this to test data loader
    # images, labels = next(iter(train_loader))
    # pic = imshow(images[0], normalize=False)

    # check input image size
    # for image, label in train_loader:
    #     print(image.shape)  # torch.Size([32, 3, 64, 64])
    #     break

    # print(dataset_train.class_to_idx) #{'cats': 0, 'dogs': 1}

    model = ConvNet().to(device)
    # train
    train(net=model, tr_loader=train_loader, te_loader=test_loader, n_epochs=100)

