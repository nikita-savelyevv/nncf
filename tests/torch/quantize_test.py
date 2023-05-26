import nncf.torch
import torch
from nncf import Dataset, quantize


shape = (1, 3, 8, 8)


class Model(torch.nn.Module):
    def forward(self, x):
        return x + torch.ones(shape)


model = Model()
quantized_model = quantize(model, Dataset([torch.zeros(shape)]))
