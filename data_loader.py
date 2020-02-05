import torch
from torch.utils.data import DataLoader, Dataset

ecg_length = 9216

class Dataset(Dataset):
    def __init__(self, featmat, label):
        self.in_data = featmat
        self.out_data = label

    def __getitem__(self, index):
        return torch.from_numpy(self.in_data[index]).float().cuda(), torch.tensor(self.out_data[index]).cuda()

    def __len__(self):
        return self.out_data.shape[0]

class Dataset_test(Dataset):
    def __init__(self, featmat, label):
        self.in_data = featmat
        self.out_data = label

    def __getitem__(self, index):
        return torch.from_numpy(self.in_data[index]).float().cuda(), self.out_data[index]

    def __len__(self):
        return self.out_data.shape[0]


def expend_length(data_in, length, move=0):
    middle = data_in.shape[2]
    right = length - middle - move
    left = move
    middle_shape = data_in.shape
    right_shape = list(data_in.shape)
    right_shape[2] = right
    left_shape = list(data_in.shape)
    left_shape[2] = left

    return np.concatenate([np.zeros(right_shape), data_in, np.zeros(left_shape)], axis=2)


def data_train():
    data_in = expend_length(np.expand_dims(x_train, 1).astype(np.float32), ecg_length)
    data_out = y_train
    return DataLoader(Dataset(data_in, data_out), batch_size=20)

def data_test():
    data_in = expend_length(np.expand_dims(x_valid, 1).astype(np.float32), ecg_length)
    data_out = y_valid
    return DataLoader(Dataset_test(data_in, data_out), batch_size=20)