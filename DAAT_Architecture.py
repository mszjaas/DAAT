
class CAM(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, in_record):
        conv_output, label_weights = in_record
        cams = []
        for i in range(conv_output.shape[0]):
            cam = label_weights.matmul(conv_output[i])
            cams.append(cam)

        return torch.stack(cams, dim=0)

#
class CamLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = torch.nn.MSELoss(reduce=True, size_average=True)

    def forward(self, cams1, cams2):
        cams1 = cams1.flip(1)
        return self.loss(cams1, cams2)


class MainModel(nn.Module):
    def __init__(self, in_model, k=0.2):
        """
        :param in_model: the model to be train
        :param k: weight of loss_{assist}
        """
        super().__init__()
        self.model = in_model()
        self.model2 = in_model()
        self.cam = CAM()
        self.camloss = CamLoss()
        self.k = k
        self.step = 1.0471 # torch.pow(10,torch.tensor([1/50]))
        self.tear_out = False

    def forward(self, in1, label, is_evaluate=False):
        if self.tear_out:
            return self.model(in1, label, is_evaluate)
        out1, loss1 = self.model(in1, label, is_evaluate)
        record1 = self.model.get_record()
        in2 = in1.flip(2)
        out2, loss2 = self.model2(in2, label, is_evaluate)
        record2 = self.model2.get_record()
        cams1 = self.cam(record1.get_input_and_weights())
        cams2 = self.cam(record2.get_input_and_weights())

        loss3 = self.camloss(cams1, cams2)
        if is_evaluate:
            return out1
        else:
            return out1, loss1 + loss2 + self.k * loss3

    def share_weight(self):
        pa = {}
        for name, param in self.model.named_parameters():
            pa[name] = param.data
        for name, param in self.model2.named_parameters():
            pa[name] = (param.data + pa[name]) / 2
        for name, param in self.model.named_parameters():
            param.data = pa[name]
        for name, param in self.model2.named_parameters():
            param.data = pa[name]