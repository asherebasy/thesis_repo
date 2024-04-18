from torch import nn


class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()

    def forward(self, inputs, targets):
        loss = self.loss(inputs, targets)

        return loss


class L1Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss()

    def forward(self, inputs, targets):
        loss = self.loss(inputs, targets)

        return loss


class CombinedLoss(nn.Module):
    def __init__(self, loss1, loss2, weight1=0.5, weight2=0.5):
        super(CombinedLoss, self).__init__()
        assert sum([weight1, weight2]) == 1, f'The sum of weights has to be 1, but got {sum([weight1, weight2])}!'
        self.loss1 = loss1
        self.loss2 = loss2
        self.weight1 = weight1
        self.weight2 = weight2

    def forward(self, inputs, targets):
        # Calculate each individual loss
        loss1_value = self.loss1(inputs, targets)
        loss2_value = self.loss2(inputs, targets)

        # Combine the losses with specified weights
        combined_loss = (self.weight1 * loss1_value) + (self.weight2 * loss2_value)

        return combined_loss
