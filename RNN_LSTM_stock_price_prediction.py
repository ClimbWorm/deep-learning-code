import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR

"""
data preprocessing
model building
training
prediction
visualization
this post on Pytorch Forum discuss the shape in LSTM: https://discuss.pytorch.org/t/please-help-lstm-input-output-dimensions/89353/11
take time to see the following notebooks:
https://github.com/udacity/deep-learning-v2-pytorch/blob/master/recurrent-neural-networks/char-rnn/Character_Level_RNN_Solution.ipynb
https://github.com/pytorch/examples/blob/main/time_sequence_prediction/train.py
https://discuss.pytorch.org/t/for-beginners-do-not-use-view-or-reshape-to-swap-dimensions-of-tensors/75524
"""

dataset_train = pd.read_csv('./dataset/Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values

# feature scaling
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)

X_train = []
y_train = []
for i in range(60, 1258):
    X_train.append(training_set_scaled[i - 60:i, 0])  # previous 60 prices
    y_train.append(training_set_scaled[i, 0])  # prices on the 61 day
X_train, y_train = np.array(X_train), np.array(y_train)
# X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
y_train = torch.Tensor(y_train)
X_train = torch.from_numpy(
    X_train).float()  # (1198, 60) if not have reshape else shape: (1198, 60, 1) 1198 observations

# create test set
dataset_test = pd.read_csv('./dataset/Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis=0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)
X_test = []
y_test = []
for i in range(60, 80):
    X_test.append(inputs[i - 60:i, 0])
    y_test.append(inputs[i, 0])
X_test, y_test = np.array(X_test), np.array(y_test)
# X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
y_test = torch.Tensor(y_test)
X_test = torch.from_numpy(X_test).float()


# predicted_stock_price = regressor.predict(X_test)
# predicted_stock_price = sc.inverse_transform(predicted_stock_price)


# model building
class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm1 = nn.LSTMCell(input_size=1, hidden_size=50)  # only one observation at a time?
        self.dropout = nn.Dropout(p=0.2)
        self.lstm2 = nn.LSTMCell(input_size=50, hidden_size=50)
        self.linear = nn.Linear(in_features=50, out_features=1)  # out_features=1 because we have only predict single y

    def forward(self, x, future=0):
        """
        x: all the input data
        if future=0 means do not perform prediction
        """
        outputs = []
        # initialize
        # x.size(0): the size of the training set,1198
        # 50: hidden size
        h_t = torch.zeros(x.size(0), 50, dtype=torch.float32)  # [1198,50]
        c_t = torch.zeros(x.size(0), 50, dtype=torch.float32)  # [1198,50]
        h_t2 = torch.zeros(x.size(0), 50, dtype=torch.float32)  # [1198,50]
        c_t2 = torch.zeros(x.size(0), 50, dtype=torch.float32)  # [1198,50]
        for input_t in x.split(1, dim=1):
            # print(input_t.size())
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))  # input_t: 1198 x 1
            h_t, c_t = self.dropout(h_t), self.dropout(c_t)

            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            h_t2, c_t2 = self.dropout(h_t2), self.dropout(c_t2)

            out = self.linear(h_t2)
            # print(out.shape)
            outputs += [out]
        for i in range(future):
            # print(out.shape)
            h_t, c_t = self.lstm1(out, (h_t, c_t))  # out: 1198 x 60
            h_t, c_t = self.dropout(h_t), self.dropout(c_t)

            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            h_t2, c_t2 = self.dropout(h_t2), self.dropout(c_t2)

            out = self.linear(h_t2)
            outputs += [out]
        outputs = torch.cat(outputs, dim=1)
        return outputs[:, -1] # [1198,]


def train(net, X_train, y_train, X_test, y_test, n_epochs=10, learning_rate=1):
    loss = nn.MSELoss()
    # optimizer = torch.optim.LBFGS(rnn.parameters(), lr=learning_rate)
    optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
    for i in range(n_epochs):
        print('STEP: ', i)
        optimizer.zero_grad()
        pred = net(X_train)
        l = loss(pred, y_train)
        print('loss:', l.item())
        l.backward()
        optimizer.step()
        scheduler.step()
        # ___________LBFGS optimizer____________
        # def closure():
        #     optimizer.zero_grad()
        #     pred = net(X_train)
        #     l = loss(pred, y_train)
        #     print('loss:', l.item())
        #     l.backward()
        #     return l
        # optimizer.step(closure)
        # __________LBFGS end_______________________

        # begin to predict, no need to track gradient here
        with torch.no_grad():
            future = 20
            pred = net(X_test, future=future)
            l = loss(pred, y_test)
            print('test loss:', l.item())
    return net

if __name__ == "__main__":
    rnn = LSTM()
    optimized_model = train(rnn, X_train, y_train, X_test, y_test, n_epochs=1000)
    pred = optimized_model(X_test)
    # it's better to transform from standardized value to its original value
    pred = pred.view(-1,1).detach().numpy()
    pred=sc.inverse_transform(pred)
    true = sc.inverse_transform(y_test.view(-1,1))
    # visualize
    fig = plt.figure()
    plt.plot(pred,color='r',label="predicted")
    plt.plot(true,color='b',label="true")
    plt.legend()
    fig.savefig('stock.jpg')


