import pandas as pd
import numpy as np
import sys
import sklearn
import torch
import torch.nn as nn
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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
class CNN(torch.nn.Module):
    def __int__(self):
        super(CNN, self).__init__()
        #https://blog.csdn.net/qq_38334521/article/details/106160029
        # self.conv = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=2)
        # self.relu = nn.ReLU()
        # self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.fc1 = nn.Linear(in_features=3, out_features=3)
        # self.fc2 = nn.Linear(in_features=3, out_features=1)
        # self.sigmoid = nn.Sigmoid()

        # self.layer_module=nn.ModuleList()
        self.conv = torch.nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # self.layer_module.append(self.cov)
        #
        # self.fc = torch.nn.Sequential(
        #     nn.Linear(in_features=3, out_features=3),  # fully connected layer
        #     nn.ReLU()
        # )
        # self.layer_module.append(self.fc)
        #
        # self.out = torch.nn.Sequential(
        #     nn.Linear(in_features=3, out_features=1),
        #     nn.Sigmoid()
        # )
        # self.layer_module.append(self.out)

    def forward(self, x):
        print(x.shape)
        out = self.conv(x)
        # out = self.layer_module[0](x)
        # out = self.layer_module[0](out)
        out = out.view(-1, 3)  # reshape
        # out = self.layer_module[1](out)
        # out = self.layer_module[2](out)
        return out


# model training
# define how to train the model
def fit(net, tr_loader, te_loader, n_epochs=100, learning_rate=0.01):
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
    loss = torch.nn.BCELoss()
    print(net.parameters())
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)  # tune the learning rate, optimize the result
    iter = 0
    for _ in tqdm(range(n_epochs)):
        for i, (images, labels) in enumerate(tr_loader):
            # Load images
            images = images.requires_grad_()

            # Forward pass to get output/logits
            y_pred = net(images)  # forward

            # Calculate Loss: softmax --> cross entropy loss
            l = loss(y_pred, labels)
            # list_of_losses.append(l.item())

            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()
            # Getting gradients w.r.t. parameters
            l.backward()
            # Updating parameters
            optimizer.step()

            iter += 1

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
                    predicted = outputs > 0.5

                    # Total number of labels
                    total += label.size(0)

                    # Total correct predictions
                    correct += (predicted == label).sum()

                accuracy = 100 * correct / total

                # Print Loss
                print('Iteration: {}. Loss: {}. Accuracy: {}'.format(iter, l.item(), accuracy))

    return None  # list_of_losses


# def predict(net, test_image):
#     """
#     :param net: the trained model
#     :param test_image: test set
#     :return: the predicted value
#     """
#     res = net(X) > 0.5  # set the threshold as 0.5 in binary classification
#     return res

# model.to(device)
if __name__ == "__main__":
    # Run this to test data loader
    # images, labels = next(iter(train_loader))
    # pic = imshow(images[0], normalize=False)
    # print(dataset_train.class_to_idx) #{'cats': 0, 'dogs': 1}
    model = CNN()
    print(next(model.parameters()))
    # model = model.to(device)

    # fit(net=model, tr_loader=train_loader, te_loader=test_loader, n_epochs=100)

    # y_pred = predict(net=model, X=X_test).detach().numpy()
    # print(y_pred.sum())
    # cm = confusion_matrix(y_test, y_pred)
    # print("Confusion Matrix: {}".format(cm))
    # acc = accuracy_score(y_test, y_pred)
    # print("Accuracy: {}".format(acc))
