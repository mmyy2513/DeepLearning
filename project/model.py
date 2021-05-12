import torch.nn as nn


class ModelClass(nn.Module):
    """
    TODO: Write down your model
    """
    def __init__(self):
        super(ModelClass, self).__init__()

        self.in_dim = 28 * 28 * 3
        self.out_dim = 10

        self.linear = nn.Linear(self.in_dim, self.out_dim)

    def forward(self, x):
        x = x.view(-1, self.in_dim)
        x = self.linear(x)
        return x
