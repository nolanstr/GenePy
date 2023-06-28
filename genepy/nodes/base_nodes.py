import torch


class Constant:

    def __init__(self, value, name):
        self.value = value
        self.name = name

    def forward(self):
        return self.value

    def backward(self):
        return 0#torch.zeros

class Add:
    
    def forward(self)
