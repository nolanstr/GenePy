import torch

class Sin:

    def forward(self, agraph_evals, node1, node2):
        return torch.sin(agraph_evals[node1])

    def backward(self, agraph_evals, node1, node2):
        pass

class Cos:

    def forward(self, agraph_evals, node1, node2):
        return torch.cos(agraph_evals[node1])

    def backward(self, agraph_evals, node1, node2):
        pass

class Tan:

    def forward(self, agraph_evals, node1, node2):
        return torch.tan(agraph_evals[node1])

    def backward(self, agraph_evals, node1, node2):
        pass

class ASin:

    def forward(self, agraph_evals, node1, node2):
        return torch.asin(agraph_evals[node1])

    def backward(self, agraph_evals, node1, node2):
        pass

class ACos:

    def forward(self, agraph_evals, node1, node2):
        return torch.acos(agraph_evals[node1])

    def backward(self, agraph_evals, node1, node2):
        pass

class ATan:

    def forward(self, agraph_evals, node1, node2):
        return torch.atan(agraph_evals[node1])

    def backward(self, agraph_evals, node1, node2):
        pass
