import numpy as np
import torch

np.random.seed(2)

T = 20
L = 1000
N = 100


if __name__ == "__main__":
    # x = np.empty((N, L), 'int64')
    # x[:] = np.array(range(L)) + np.random.randint(-4 * T, 4 * T, N).reshape(N, 1)
    # data = np.sin(x / 1.0 / T).astype('float64')
    # torch.save(data, open('traindata.pt', 'wb'))

    data = torch.load('traindata.pt')
    input = torch.from_numpy(data[3:, :-1])
    print(data.shape)
    print(input.shape)
    print(input.size(0))
    print(input.split(1, dim=1)[0].shape) # 97 x 1