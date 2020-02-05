class first_block(nn.Module):
    def __init__(self):
        super().__init__()
        self.x1 = nn.Conv1d(64, 64, 15, padding=7)
        self.x2 = nn.BatchNorm1d(64)
        self.x3 = nn.ReLU()
        self.x4 = nn.Dropout(0.5)
        self.x5 = nn.Conv1d(64, 64, 15, padding=7)
        self.x6 = nn.MaxPool1d(2, stride=2)

        self.y1 = nn.MaxPool1d(2, stride=2)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight.data)

    def forward(self, in_array):
        x = self.x1(in_array)
        x = self.x2(x)
        x = self.x3(x)
        x = self.x4(x)
        x = self.x5(x)
        x = self.x6(x)

        y = self.y1(in_array)
        return torch.add(x, y)


class res_block(nn.Module):
    def __init__(self, in_channel, out_chaneel, has_conv, has_pool):
        super().__init__()
        self.has_conv = has_conv
        self.has_pool = has_pool
        self.x1 = nn.BatchNorm1d(in_channel)
        self.x2 = nn.ReLU()
        self.x3 = nn.Dropout(0.5)
        self.x4 = nn.Conv1d(in_channel, out_chaneel, 15, padding=7)
        self.x5 = nn.BatchNorm1d(out_chaneel)
        self.x6 = nn.ReLU()
        self.x7 = nn.Dropout(0.5)
        self.x8 = nn.Conv1d(out_chaneel, out_chaneel, 15, padding=7)
        self.pool_x = nn.MaxPool1d(2, 2)

        self.conv_y = nn.Conv1d(in_channel, out_chaneel, kernel_size=1)
        self.pool_y = nn.MaxPool1d(2, 2)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight.data)

    def forward(self, in_array):
        x = self.x1(in_array)
        x = self.x2(x)
        x = self.x3(x)
        x = self.x4(x)
        x = self.x5(x)
        x = self.x6(x)
        x = self.x7(x)
        x = self.x8(x)
        if self.has_conv:
            y = self.conv_y(in_array)
        else:
            y = in_array
        if self.has_pool:
            x = self.pool_x(x)
            y = self.pool_y(y)
        return torch.add(x, y)


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.x1 = nn.Conv1d(1, 64, 15, padding=7)
        self.x2 = nn.BatchNorm1d(64)
        self.x3 = nn.ReLU()
        self.x4 = first_block()

        self.x5 = res_block(64, 64, False, False)
        self.x6 = res_block(64, 64, False, True)
        self.x7 = res_block(64, 64, False, False)
        self.x8 = res_block(64, 64, False, True)

        self.x9 = res_block(64, 128, True, False)
        self.x10 = res_block(128, 128, False, True)
        self.x11 = res_block(128, 128, False, False)
        self.x12 = res_block(128, 128, False, True)

        self.x13 = res_block(128, 256, True, False)
        self.x14 = res_block(256, 256, False, True)
        self.x15 = res_block(256, 256, False, False)
        self.x16 = res_block(256, 256, False, True)

        self.x17 = res_block(256, 512, True, False)
        self.x18 = res_block(512, 512, False, True)
        # self.x19 = res_block(512, 512, False, False)
        # self.x20 = res_block(512, 512, False, True)

        self.x21 = nn.BatchNorm1d(512)
        self.x22 = nn.ReLU()
        self.classify0 = torch.nn.AdaptiveAvgPool1d(1)
        self.classify1 = nn.Linear(512, 4)
        self.classify2 = nn.Softmax()
        nn.init.kaiming_normal_(self.classify1.weight.data)

        self.loss = torch.nn.CrossEntropyLoss()
        self.record = Recorder()

    def forward(self, in_array, label, is_evaluate=False):
        x = self.x1(in_array)
        x = self.x2(x)
        x = self.x3(x)
        x = self.x4(x)
        x = self.x5(x)
        x = self.x6(x)
        x = self.x7(x)
        x = self.x8(x)
        x = self.x9(x)
        x = self.x10(x)
        x = self.x11(x)
        x = self.x12(x)
        x = self.x13(x)
        x = self.x14(x)
        x = self.x15(x)
        x = self.x16(x)
        x = self.x17(x)
        x = self.x18(x)
        # x = self.x19(x)
        # x = self.x20(x)
        x = self.x21(x)
        x = self.x22(x)
        self.record.set_input(x)
        self.record.set_weights(self.classify1.weight)

        x = self.classify0(x)
        x = torch.flatten(x, 1)
        x = self.classify1(x)
        x = self.classify2(x)
        if is_evaluate:
            return x
        return x, self.loss(x, label)

    def get_record(self):
        return self.record