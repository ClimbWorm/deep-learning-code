import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

# the working directory is:
# /Users/Shared/Files From d.localized/Udemy/Machine Learning A-Z (Codes and Datasets)/Part 8 - Deep Learning
path = "./AutoEncoders/"
movies = pd.read_csv(path + "ml-1m/movies.dat", sep="::", header=None, engine='python', encoding='latin-1')
users = pd.read_csv(path + "ml-1m/users.dat", sep="::", header=None, engine='python', encoding='latin-1')
ratings = pd.read_csv(path + "ml-1m/ratings.dat", sep="::", header=None, engine='python', encoding='latin-1')

training_set = pd.read_csv(path + "ml-100k/u1.base", delimiter='\t')  # it's a 8:2 training test split
# in order to apply pytorch, we should transform the data frame into array
training_set = np.array(training_set, dtype='int')
# in the training and test set, the movies won't be the same, but the users are the same
test_set = pd.read_csv(path + "ml-100k/u1.test", delimiter='\t')  # it's a 8:2 training test split
test_set = np.array(test_set, dtype='int')

# find the max id of users and movies
nb_users = int(max(max(training_set[:, 0]), max(test_set[:, 0])))
nb_movies = int(max(max(training_set[:, 1]), max(test_set[:, 1])))


# create matrix for both training set and test set: user in rows and movie in columns
# same as observation as rows and features as columns
def convert(data):
    """
    :param data: training data or test data
    :return: list of lists, which is torch expect
    """
    new_data = []
    for id_users in range(1, nb_users + 1):
        id_movies = data[:, 1][data[:, 0] == id_users]
        id_ratings = data[:, 2][data[:, 0] == id_users]
        # if the user rate the movie, replace the rating with the real rating, else use 0
        rating = np.zeros(nb_movies)
        rating[id_movies - 1] = id_ratings
        new_data.append(list(rating))
    return new_data


training_set = convert(training_set)
test_set = convert(test_set)

# tensor can only have single type
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)


# ------------------------The above code is the same as in Boltzmann Machine------------------------
# stacked auto encoders
class SAE(nn.Module):
    def __init__(self):
        super(SAE, self).__init__()
        # encoding layer
        self.fc1 = nn.Linear(in_features=nb_movies, out_features=20)  # first fully connected layer
        self.fc2 = nn.Linear(in_features=20, out_features=10)
        # decoding layer (be symmetric)
        self.fc3 = nn.Linear(in_features=10, out_features=20)
        self.fc4 = nn.Linear(in_features=20, out_features=nb_movies)

        self.activation = nn.Sigmoid()

    def forward(self, x):
        """
        :param x: the dimension should be (1,nb_movies)
        :return:
        """
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)  # don't need activation in the last output layer
        return x


if __name__ == "__main__":
    sae = SAE()
    criterion = nn.MSELoss()
    optimizer = optim.RMSprop(sae.parameters(), lr=0.01, weight_decay=0.5)

    # training the SAE
    nb_epoch = 200
    for epoch in range(1, nb_epoch + 1):
        train_loss = 0
        s = 0.  # counter the number of users rated at least 1 movie, set it to float because RMSE requires float
        for id_user in range(nb_users):  # trained one observation a time, or Online Learning
            # because pytorch cannot except a single vector of 1-d,
            # we use Variable and then use unsqueeze to increase a dimension at position 0 (create a batch)
            input = Variable(training_set[id_user]).unsqueeze(0)
            target = input.clone()
            target.require_grad = False  # because we only need grad of input, so do this to save unnecessary work
            if torch.sum(target.data > 0) > 0:  # user rated at least 1 movie
                output = sae.forward(input)
                output[target == 0] = 0
                loss = criterion(output, target)
                mean_corrector = nb_movies / float(torch.sum(target.data > 0) + 1e-10)
                loss.backward()  # decide the direction to which the weight would be updated
                train_loss += np.sqrt(loss.data * mean_corrector)  # RMSE
                s += 1.
                optimizer.step()  # decide the intensity of the update
        print("At Epoch {}, the training loss: {}".format(epoch, train_loss / s)) # 0.9120

    # testing the SAE

    test_loss = 0
    s = 0.
    for id_user in range(nb_users):
        input = Variable(training_set[id_user]).unsqueeze(0)  # keep the training_set here
        target = Variable(test_set[id_user]).unsqueeze(0)
        target.require_grad = False  # because we only need grad of input, so do this to save unnecessary work
        if torch.sum(target.data > 0) > 0:  # user rated at least 1 movie
            output = sae.forward(input)
            output[target == 0] = 0
            loss = criterion(output, target)
            mean_corrector = nb_movies / float(torch.sum(target.data > 0) + 1e-10)
            test_loss += np.sqrt(loss.data * mean_corrector)
            s += 1.
    print("The test loss: {}".format(test_loss / s)) #0.9466
