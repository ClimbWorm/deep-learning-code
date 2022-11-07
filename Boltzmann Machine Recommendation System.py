import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from typing import List, Dict

# the working directory is:
# /Users/Shared/Files From d.localized/Udemy/Machine Learning A-Z (Codes and Datasets)/Part 8 - Deep Learning
path = "./Boltzmann_Machines/"
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


# convert the ratings into binary ratings: 1(liked),0(not like)
def binary_ratings(data):
    data[data == 0] = -1  # temporary convert original 0: not rated to -1
    data[data == 1] = 0  # cannot write data==1 or 0, because it doesn't work for tensor
    data[data == 2] = 0
    data[data >= 3] = 1
    return data


training_set = binary_ratings(training_set)
test_set = binary_ratings(test_set)


# create restricted Boltzmann Machine (RBM): a probability graphical model/energy-based model
class RBM():  # Bernoulli RBM (binary)
    def __init__(self, nv, nh):
        """
        :param nv: number of visible nodes
        :param nh: number of hidden nodes
        """
        # initialize weights
        self.W = torch.randn(nh, nv)
        # initialize bias
        self.a = torch.randn(1, nh)  # p(h|v) create fake dimension for batch in the first param
        self.b = torch.randn(1, nv)  # p(v|h)

    def sample_h(self, x):
        """
        :param x: denotes v in p(h=1|v)
        :return: sample of hidden neurons according to the p(h|v)
        """
        # sampling the hidden nodes according to p(h=1|v), it's a sigmoid function
        # use git sampling to approximate MLE
        wx = torch.mm(x, self.W.t())  # matrix multiplication
        activation = wx + self.a.expand_as(
            wx)  # expand_as to make sure the bias is applied to each line of the mini batch
        p_h_given_v = torch.sigmoid(activation)
        # bernoulli sampling: if the node get value 0.7 after the activation function, then randomly generate number
        # between 0 and 1, if the number < 0.7, assign 1 else assign 0.
        return p_h_given_v, torch.bernoulli(p_h_given_v)

    def sample_v(self, y):
        """
        :param y: hidden nodes
        :return: sample of visible neurons according to the p(v|h)
        """
        # sampling the visible nodes according to p(v=1|h)
        wy = torch.mm(y, self.W)
        activation = wy + self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)

    # contrastive divergence: approximate gradient with Gibbs sampling
    # based on paper: "An Introduction to Restricted Boltzmann Machines" by Asja Fischer and Christian Igel
    def train(self, v0, vk, ph0, phk):
        """
        :param v0: input value of observations
        :param vk: observations after k step
        :param ph0: p(h0|v0)
        :param phk: p(hk|vk)
        """
        self.W += torch.mm(ph0.t(), v0) - torch.mm(phk.t(), vk)
        # because we train in batches, so we accumulate bias using torch.sum()
        self.b += torch.sum((v0 - vk), 0)  # 0: axis,torch.sum() dealing with bias updates also showed in HW3 of stat430
        self.a += torch.sum((ph0 - phk), 0)


if __name__ == "__main__":
    print(movies.head())
    nv = len(
        training_set[0])  # or nv = nb_movies, because it's a binary prediction, so each movie should use one neuron
    nh = 100  # choose any number we like as features, tunable
    batch_size = 100  # if 1: online learning
    rbm = RBM(nv, nh)

    # training
    nb_epoch = 10
    k_step = 10
    for epoch in range(1, nb_epoch + 1):
        train_loss = 0
        s = 0.  # counter, for the purpose of normalize
        for id_user in range(0, nb_users - batch_size + 1, batch_size):
            # first batch from user 0 to 99
            # second batch from user 100 to 199
            # last batch from user nb_users-99-1 to nb_users-1 (because idx starts from 0)
            v0 = training_set[id_user:id_user + batch_size]
            vk = training_set[id_user:id_user + batch_size]
            ph0, _ = rbm.sample_h(v0)

            for k in range(k_step):  # MCMC (monte carlo markov chain) technique, Gibbs sampling
                # write vk instead of v0, because we don't want to change target(v0) that
                # we will need later to evaluate our results
                _, hk = rbm.sample_h(vk)
                _, vk = rbm.sample_v(hk)
                vk[v0 < 0] = v0[v0 < 0]  # the originally missing ratings which we assigned as -1
            phk, _ = rbm.sample_h(vk)
            rbm.train(v0, vk, ph0, phk)
            train_loss += torch.mean(torch.abs(v0[v0 >= 0] - vk[v0 >= 0]))
            s += 1.
            print("At Epoch {}, the training loss: {}".format(epoch, train_loss / s))

    # testing
    test_loss = 0
    s = 0.  # counter, for the purpose of normalize
    for id_user in range(nb_users):
        vt = test_set[id_user:id_user + 1]  # vt means target
        # using the input of the training set to activate the neurons
        # of the rbm to make prediction
        v = training_set[id_user:id_user + 1]

        if len(vt[vt >= 0]) > 0:
            # only need 1 more step to take prediction on testing set
            _, h = rbm.sample_h(v)
            _, v = rbm.sample_v(h)
            test_loss += torch.mean(torch.abs(vt[vt >= 0] - v[vt >= 0]))
            s += 1.
    print("The test loss: {}".format(test_loss / s))
