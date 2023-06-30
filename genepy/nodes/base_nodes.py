import torch

class Var:

    def forward(_x, agraph_evals, agraph_consts, node1, node2):
        with torch.enable_grad():
            _var = _x[None, :, node1, None]
        return _var
    
    def _string(agraph_strings, node1, node2):
        return f"X_{node1}"

class Const:

    def forward(_x, agraph_evals, agraph_consts, node1, node2):
        with torch.enable_grad():
            _consts = agraph_consts[:, node1, None, None]
        return _consts

    def _string(agraph_strings, node1, node2):
        return f"C_{node1}"

class IConst:

    def forward(_x, agraph_evals, agraph_consts, node1, node2):
        return torch.tensor([[node1]], requires_grad=True)

    def _string(agraph_strings, node1, node2):
        if isinstance(node1, float):
            return f"I_(1/{int(1/node1)})"
        return f"I_{node1}"

class Add:

    def forward(_x, agraph_evals, agraph_consts, node1, node2):
        return torch.add(agraph_evals[node1], agraph_evals[node2])

    def _string(agraph_strings, node1, node2):
        return f"({agraph_strings[node1]} + {agraph_strings[node2]})"

class Sub:

    def forward(_x, agraph_evals, agraph_consts, node1, node2):
        return torch.subtract(agraph_evals[node1], agraph_evals[node2])

    def _string(agraph_strings, node1, node2):
        return f"({agraph_strings[node1]} - {agraph_strings[node2]})"

class Mult:

    def forward(_x, agraph_evals, agraph_consts, node1, node2):
        return torch.multiply(agraph_evals[node1], agraph_evals[node2])

    def _string(agraph_strings, node1, node2):
        return f"({agraph_strings[node1]} * {agraph_strings[node2]})"

class Div:

    def forward(_x, agraph_evals, agraph_consts, node1, node2):
        return torch.divide(agraph_evals[node1], agraph_evals[node2])

    def _string(agraph_strings, node1, node2):
        return f"({agraph_strings[node1]} * {agraph_strings[node2]})"

class Pow:

    def forward(_x, agraph_evals, agraph_consts, node1, node2):
        return torch.pow(agraph_evals[node1], agraph_evals[node2])

    def _string(agraph_strings, node1, node2):
        return f"({agraph_strings[node1]})^({agraph_strings[node2]})"

class Sqrt:

    def forward(_x, agraph_evals, agraph_consts, node1, node2):
        return torch.sqrt(agraph_evals[node1])

    def _string(agraph_strings, node1, node2):
        return f"sqrt({agraph_strings[node1]})"

class Exp:

    def forward(_x, agraph_evals, agraph_consts, node1, node2):
        return torch.exp(agraph_evals[node1])

    def _string(agraph_strings, node1, node2):
        return f"exp({agraph_strings[node1]})"

class Log:

    def forward(_x, agraph_evals, agraph_consts, node1, node2):
        return torch.log(agraph_evals[node1])

    def _string(agraph_strings, node1, node2):
        return f"log({agraph_strings[node1]})"
