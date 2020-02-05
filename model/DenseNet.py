class dense_layer(nn.Module):
    def __init__(self, channel_in, kernel_size, stride=1):
        super().__init__()
        self.x1 = nn.ReLU()
        self.x2 = nn.Conv1d(channel_in, channel_in, kernel_size, padding=int((kernel_size - 1) / 2), bias=False,
                            stride=stride)
        self.x3 = nn.Dropout(0.05)
        nn.init.kaiming_normal_(self.x2.weight.data)

    def forward(self, array_in):
        x = self.x1(array_in)
        x = self.x2(x)
        x = self.x3(x)
        return x


class dense_block(nn.Module):
    def __init__(self, channel_in, kernel_size):
        super().__init__()
        self.x1 = dense_layer(channel_in, kernel_size, stride=2)
        self.x2 = dense_layer(channel_in, kernel_size)
        self.x3 = dense_layer(channel_in, kernel_size)
        self.x4 = dense_layer(channel_in, kernel_size)

    def forward(self, array_in):
        x1 = self.x1(array_in)
        x2 = self.x2(x1)
        x3 = self.x3(x2)
        x4 = self.x4(x3)
        return torch.cat([x1, x2, x3, x4], dim=1)


class transition_block(nn.Module):
    def __init__(self, channel_in):
        super().__init__()
        self.x1 = nn.Conv1d(channel_in, channel_in, 1)
        self.x2 = nn.Dropout(0.05)
        self.x3 = nn.AvgPool1d(2)
        self.x4 = nn.BatchNorm1d(channel_in)
        nn.init.kaiming_normal_(self.x1.weight.data)

    def forward(self, array_in):
        x = self.x1(array_in)
        x = self.x2(x)
        x = self.x3(x)
        x = self.x4(x)
        return x


class DenseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.x1 = nn.Conv1d(1, 16, 21, padding=10, bias=False)
        self.x2 = nn.BatchNorm1d(16)
        self.x3 = dense_block(16, 15)
        self.x4 = transition_block(16 * 4)
        self.x5 = dense_block(64, 7)
        self.x6 = transition_block(64 * 4)
        self.x7 = dense_block(256, 5)
        self.x8 = nn.ReLU()
        self.x9 = nn.AdaptiveAvgPool1d(1)
        self.classify1 = nn.Linear(1024, 4)
        self.classify2 = nn.Softmax()
        self.loss = torch.nn.CrossEntropyLoss()
        self.record = Recorder()

        nn.init.kaiming_normal_(self.x1.weight.data)
        nn.init.kaiming_normal_(self.classify1.weight.data)

    def forward(self, array_in, label, is_evaluate=False):
        x = self.x1(array_in)
        x = self.x2(x)
        x = self.x3(x)
        x = self.x4(x)
        x = self.x5(x)
        x = self.x6(x)
        x = self.x7(x)
        x = self.x8(x)
        self.record.set_input(x)
        x = self.x9(x)
        x = torch.flatten(x, 1)
        x = self.classify1(x)
        x = self.classify2(x)
        self.record.set_weights(self.classify1.weight)
        if is_evaluate:
            return x
        return x, self.loss(x, label)

    def get_record(self):
        return self.record