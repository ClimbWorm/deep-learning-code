import pandas as pd
# import numpy as np
from tqdm import tqdm
import torch
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

"""
The pipeline of this churn prediction classification problem based on Artificial Neural Network:
data cleaning
train_test_split
feature scaling
model building
model training
model evaluation
prediction
"""

# import dataset
# columns: ['RowNumber', 'CustomerId', 'Surname', 'CreditScore', 'Geography',
#        'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',
#        'IsActiveMember', 'EstimatedSalary', 'Exited']
data = pd.read_csv("Churn_Modelling.csv")
X = data.iloc[:, 3:-1].values  # we dismiss some useless variables
y = data.iloc[:, -1].values

# after selection, the columns:
# ['CreditScore', 'Geography',
#  'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',
#  'IsActiveMember', 'EstimatedSalary', 'Exited']

# encode the variable:
# do label encoding on "Gender"
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])

# do one-hot encoding on "Geography"
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])],
                       remainder='passthrough')  # transformers: (whatever name,transformer,columns)
X = ct.fit_transform(X)

# split training and testing data (this step should before feature scaling)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=16)

# feature scaling
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)  # only transform, no fit for sake of information leakage

# change the type to tensor(because the model requires tensors)
# be sure to change to tensor type and change to float!
X_train = torch.from_numpy(X_train).float()
X_test = torch.from_numpy(X_test).float()
y_train, y_test = torch.from_numpy(y_train).float(), torch.from_numpy(y_test).float()

# building artificial neural network model
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


# define the architecture of nn
class ANN(torch.nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        self.n_in = X_train.shape[1]  # tensors are row major in pytorch
        self.net = torch.nn.Sequential(
            torch.nn.Linear(self.n_in, 6),
            torch.nn.ReLU(),
            torch.nn.Linear(6, 6),
            torch.nn.ReLU(),
            torch.nn.Linear(6, 1),  # 1 here because this is a binary classification problem, only 1 output is enough
            torch.nn.Sigmoid()  # in multi-class scenario, we should use softmax layer
        )  # torch.nn.Linear same as tf.keras.layers.Dense(): fully connected dense layer

    def forward(self, x):  # model()会自动调用forward，不需要model.forward()
        out = self.net(x).flatten()  # x: X_train
        return out


# define how to train the model
def fit(net, X, y, n_epochs=100):
    """
    :param net: the architecture we built before, e.g.ANN(torch.nn.Module)
    :param X: X_train, n x d tensor
    :param y: y_train, n x 1
    :param n_epochs: training epochs
    :return: list of losses
    """
    list_of_losses = []
    # loss = torch.nn.CrossEntropyLoss() # this is for multi-class classification
    loss = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters())  # ,lr=)

    for epoch in tqdm(range(n_epochs)):
        y_pred = net.forward(X)
        l = loss(y_pred, y)  # should put y_pred in first pram
        list_of_losses.append(l.item())
        l.backward()
        optimizer.step()  # function as updating parameters, param -= learning_rate * param.grad
        optimizer.zero_grad()
        if epoch % 10 == 0:
            # print(next(net.parameters()))
            # print(next(net.parameters()))
            # [w, b] = net.parameters()
            # print(f'epoch {epoch + 1}: w={w[0][0].item():.3f},loss={l:.8f}')
            print("epoch {}: loss={}".format(epoch + 1, l))
    return list_of_losses


def predict(net, X):
    """
    :param net: the trained model
    :param X: test set
    :return: the predicted value
    """
    res = net(X) > 0.5  # set the threshold as 0.5 in binary classification
    return res


if __name__ == "__main__":
    # print(np.asarray(y_train).shape)
    model = ANN()
    losses = fit(net=model, X=X_train, y=y_train, n_epochs=100)
    y_pred = predict(net=model,X = X_test).detach().numpy()
    print(y_pred.sum())
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix: {}".format(cm))
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy: {}".format(acc))
