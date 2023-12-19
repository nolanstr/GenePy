import torch

class Var:
    def forward(_x, agraph_evals, agraph_consts, node1, node2):
        """
        Parameters
        ----------
        _x : [Argument]
        agraph_evals : [Argument]
        agraph_consts : [Argument]
        node1 : [Argument]
        node2 : [Argument]

        """
        with torch.enable_grad():
            _var = _x[None, :, node1, None]
        return _var
    
    def _string(agraph_strings, node1, node2):
        """
        Parameters
        ----------
        agraph_strings : [Argument]
        node1 : [Argument]
        node2 : [Argument]

        """
        return f"X_{node1}"

class Const:

    def forward(_x, agraph_evals, agraph_consts, node1, node2):
        """
        Parameters
        ----------
        _x : [Argument]
        agraph_evals : [Argument]
        agraph_consts : [Argument]
        node1 : [Argument]
        node2 : [Argument]

        """
        with torch.enable_grad():
            _consts = agraph_consts[:, node1, None, None]
        return _consts

    def _string(agraph_strings, node1, node2):
        """
        Parameters
        ----------
        agraph_strings : [Argument]
        node1 : [Argument]
        node2 : [Argument]

        """
        return f"C_{node1}"

class IConst:

    def forward(_x, agraph_evals, agraph_consts, node1, node2):
        """
        Changed from true as constants cant have requires grad
        
        Parameters
        ----------
        _x : [Argument]
        agraph_evals : [Argument]
        agraph_consts : [Argument]
        node1 : [Argument]
        node2 : [Argument]

        """
        return torch.tensor([[node1]], requires_grad=False)

    def _string(agraph_strings, node1, node2):
        """
        Parameters
        ----------
        agraph_strings : [Argument]
        node1 : [Argument]
        node2 : [Argument]

        """
        if isinstance(node1, float):
            return f"(1/{int(1/node1)})"
        return f"{node1}"

class Add:
    def forward(_x, agraph_evals, agraph_consts, node1, node2):
        """
        Parameters
        ----------
        _x : [Argument]
        agraph_evals : [Argument]
        agraph_consts : [Argument]
        node1 : [Argument]
        node2 : [Argument]

        """
        return torch.add(agraph_evals[node1], agraph_evals[node2])

    def _string(agraph_strings, node1, node2):
        """
        Parameters
        ----------
        agraph_strings : [Argument]
        node1 : [Argument]
        node2 : [Argument]

        """
        return f"({agraph_strings[node1]} + {agraph_strings[node2]})"

class Sub:
    def forward(_x, agraph_evals, agraph_consts, node1, node2):
        """
        Parameters
        ----------
        _x : [Argument]
        agraph_evals : [Argument]
        agraph_consts : [Argument]
        node1 : [Argument]
        node2 : [Argument]

        """
        return torch.subtract(agraph_evals[node1], agraph_evals[node2])

    def _string(agraph_strings, node1, node2):
        """
        Parameters
        ----------
        agraph_strings : [Argument]
        node1 : [Argument]
        node2 : [Argument]

        """
        return f"({agraph_strings[node1]} - {agraph_strings[node2]})"

class Mult:

    def forward(_x, agraph_evals, agraph_consts, node1, node2):
        """
        Parameters
        ----------
        _x : [Argument]
        agraph_evals : [Argument]
        agraph_consts : [Argument]
        node1 : [Argument]
        node2 : [Argument]

        """
        return torch.multiply(agraph_evals[node1], agraph_evals[node2])

    def _string(agraph_strings, node1, node2):
        """
        Parameters
        ----------
        agraph_strings : [Argument]
        node1 : [Argument]
        node2 : [Argument]

        """
        return f"({agraph_strings[node1]} * {agraph_strings[node2]})"

class Div:
    def forward(_x, agraph_evals, agraph_consts, node1, node2):
        """
        Parameters
        ----------
        _x : [Argument]
        agraph_evals : [Argument]
        agraph_consts : [Argument]
        node1 : [Argument]
        node2 : [Argument]

        """
        return torch.divide(agraph_evals[node1], agraph_evals[node2])

    def _string(agraph_strings, node1, node2):
        """
        Parameters
        ----------
        agraph_strings : [Argument]
        node1 : [Argument]
        node2 : [Argument]

        """
        return f"({agraph_strings[node1]} * {agraph_strings[node2]})"

class Pow:
    def forward(_x, agraph_evals, agraph_consts, node1, node2):
        """
        Parameters
        ----------
        _x : [Argument]
        agraph_evals : [Argument]
        agraph_consts : [Argument]
        node1 : [Argument]
        node2 : [Argument]

        """
        return torch.pow(agraph_evals[node1], agraph_evals[node2])

    def _string(agraph_strings, node1, node2):
        """
        Parameters
        ----------
        agraph_strings : [Argument]
        node1 : [Argument]
        node2 : [Argument]

        """
        return f"({agraph_strings[node1]})^({agraph_strings[node2]})"

class Square:

    def forward(_x, agraph_evals, agraph_consts, node1, node2):
        """
        Parameters
        ----------
        _x : [Argument]
        agraph_evals : [Argument]
        agraph_consts : [Argument]
        node1 : [Argument]
        node2 : [Argument]

        """
        return torch.square(agraph_evals[node1])

    def _string(agraph_strings, node1, node2):
        """
        Parameters
        ----------
        agraph_strings : [Argument]
        node1 : [Argument]
        node2 : [Argument]

        """
        return f"({agraph_strings[node1]})^(2)"

class Sqrt:

    def forward(_x, agraph_evals, agraph_consts, node1, node2):
        """
        Parameters
        ----------
        _x : [Argument]
        agraph_evals : [Argument]
        agraph_consts : [Argument]
        node1 : [Argument]
        node2 : [Argument]

        """
        return torch.sqrt(agraph_evals[node1])

    def _string(agraph_strings, node1, node2):
        """
        Parameters
        ----------
        agraph_strings : [Argument]
        node1 : [Argument]
        node2 : [Argument]

        """
        return f"sqrt({agraph_strings[node1]})"

class Exp:
    def forward(_x, agraph_evals, agraph_consts, node1, node2):
        """
        Parameters
        ----------
        _x : [Argument]
        agraph_evals : [Argument]
        agraph_consts : [Argument]
        node1 : [Argument]
        node2 : [Argument]

        """
        return torch.exp(agraph_evals[node1])

    def _string(agraph_strings, node1, node2):
        """
        Parameters
        ----------
        agraph_strings : [Argument]
        node1 : [Argument]
        node2 : [Argument]

        """
        return f"exp({agraph_strings[node1]})"

class Log:
    def forward(_x, agraph_evals, agraph_consts, node1, node2):
        """
        Parameters
        ----------
        _x : [Argument]
        agraph_evals : [Argument]
        agraph_consts : [Argument]
        node1 : [Argument]
        node2 : [Argument]

        """
        return torch.log(agraph_evals[node1])

    def _string(agraph_strings, node1, node2):
        """
        Parameters
        ----------
        agraph_strings : [Argument]
        node1 : [Argument]
        node2 : [Argument]

        """
        return f"log({agraph_strings[node1]})"
