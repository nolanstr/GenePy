import torch

class Sin:

    def forward(_x, agraph_evals, agraph_consts, node1, node2):
        return torch.sin(agraph_evals[node1])

    def _string(agraph_strings, node1, node2):
        return f"sin({agraph_strings[node1]})"

class Cos:

    def forward(_x, agraph_evals, agraph_consts, node1, node2):
        return torch.cos(agraph_evals[node1])

    def _string(agraph_strings, node1, node2):
        return f"cos({agraph_strings[node1]})"

class Tan:

    def forward(_x, agraph_evals, agraph_consts, node1, node2):
        return torch.tan(agraph_evals[node1])

    def _string(agraph_strings, node1, node2):
        return f"tan({agraph_strings[node1]})"

class ASin:

    def forward(_x, agraph_evals, agraph_consts, node1, node2):
        return torch.asin(agraph_evals[node1])

    def _string(agraph_strings, node1, node2):
        return f"asin({agraph_strings[node1]})"

class ACos:

    def forward(_x, agraph_evals, agraph_consts, node1, node2):
        return torch.acos(agraph_evals[node1])

    def _string(agraph_strings, node1, node2):
        return f"acos({agraph_strings[node1]})"

class ATan:

    def forward(_x, agraph_evals, agraph_consts, node1, node2):
        return torch.atan(agraph_evals[node1])

    def _string(agraph_strings, node1, node2):
        return f"atan({agraph_strings[node1]})"

class Sinh:

    def forward(_x, agraph_evals, agraph_consts, node1, node2):
        return torch.sinh(agraph_evals[node1])

    def _string(agraph_strings, node1, node2):
        return f"sinh({agraph_strings[node1]})"

class Cosh:

    def forward(_x, agraph_evals, agraph_consts, node1, node2):
        return torch.cosh(agraph_evals[node1])

    def _string(agraph_strings, node1, node2):
        return f"cosh({agraph_strings[node1]})"

class Tanh:

    def forward(_x, agraph_evals, agraph_consts, node1, node2):
        return torch.tanh(agraph_evals[node1])

    def _string(agraph_strings, node1, node2):
        return f"tanh({agraph_strings[node1]})"
