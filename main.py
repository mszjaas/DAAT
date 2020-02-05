
import numpy as np
import torch
import numpy as np
import torch
from torch import nn, optim
from sklearn.metrics import confusion_matrix
from DAAT_Architecture import MainModel
from data_loader import data_train, data_test

from model.ResNet import ResNet
from model.DenseNet import DenseNet
from model.SENet import SENet


# load data
x_train = np.load('x_train.npy').astype(np.float32)
x_valid = np.load('x_valid.npy').astype(np.float32)
y_train = np.load('y_train.npy').astype(np.int64)
y_valid = np.load('y_valid.npy').astype(np.int64)



# evaluate
def evaluate_model(model, epoch, name, last_accuracy):
    def write_log(a):
        with open('name' + '.log', 'a+') as f:
            f.write(str(a))

    data = data_test()
    pred = np.array([])
    real = np.array([])
    for i, o in data:
        if model.tear_out:
            x = model(i, torch.flatten(o), True)
        else:
            x = model.model(i, torch.flatten(o), True)
        x = np.argmax(x.cpu().detach().numpy(), 1)
        pred = np.hstack((pred, x))
        real = np.hstack((real, o))
    cvconfusion = confusion_matrix(real, pred)
    right = cvconfusion[0, 0] + cvconfusion[1, 1] + cvconfusion[2, 2] + cvconfusion[3, 3]
    all = np.sum(cvconfusion)
    accuracy = round(right / all, 4)
    row_sum = np.sum(cvconfusion, 1)
    col_sum = np.sum(cvconfusion, 0)
    f1_1 = round(2 * cvconfusion[0, 0] / (row_sum[0] + col_sum[0]), 4)
    f1_2 = round(2 * cvconfusion[1, 1] / (row_sum[1] + col_sum[1]), 2)
    f1_3 = round(2 * cvconfusion[2, 2] / (row_sum[2] + col_sum[2]), 2)
    f1_4 = round(2 * cvconfusion[3, 3] / (row_sum[3] + col_sum[3]), 2)
    report = 'epoch:\t{}\taccuracy:\t{}\tf1:\t{}\t{}\t{}\t{}'.format(epoch, accuracy, f1_1, f1_2, f1_3, f1_4)
    print(report)
    write_log(report + '\n')
    if accuracy > last_accuracy:
        torch.save(model.state_dict(), name)
        last_val = accuracy
    else:
        model.load_state_dict(torch.load(name))

    return accuracy



# train
def train(train_data, model, epochs):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    #     optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
    progress = tqdm(range(epochs))
    accuracy = 0
    for epoch in progress:
        batch = 0
        for batch_in, batch_out in train_data:
            batch += 1

            progress.set_description('epoch: {}   batch: {} '.format(epoch, batch))
            #             optimizer.zero_grad()

            x, loss = model(batch_in, torch.flatten(batch_out), False)
            optimizer.zero_grad()
            #             print(output.cpu())
            #             print(batch_out.flatten().cpu())
            loss.backward()
            optimizer.step()
            model.share_weight()
        accuracy = evaluate_model(model, epoch, accuracy)
        if epoch < 200:
            model.lamda /= model.step
        else:
            model.tear_out = True


# entry, results will be in .log

# ResNet
model = MainModel(ResNet, 0.1).cuda()
evaluate_model(model, 0)
train_data = data_train()
train(train_data, model, 300)

# DenseNet
model = MainModel(DenseNet, 0.001).cuda()
evaluate_model(model, 0)
train_data = data_train()
train(train_data, model, 300)

# SENet
model = MainModel(SENet, 1.0).cuda()
evaluate_model(model, 0)
train_data = data_train()
train(train_data, model, 300)