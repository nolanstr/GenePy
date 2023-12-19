import torch

class Sin:
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
        return torch.sin(agraph_evals[node1])

    def _string(agraph_strings, node1, node2):
        """
        Parameters
        ----------
        agraph_strings : [Argument]
        node1 : [Argument]
        node2 : [Argument]

        """
        return f"sin({agraph_strings[node1]})"

class Cos:
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
        return torch.cos(agraph_evals[node1])

    def _string(agraph_strings, node1, node2):
        """
        Parameters
        ----------
        agraph_strings : [Argument]
        node1 : [Argument]
        node2 : [Argument]

        """
        return f"cos({agraph_strings[node1]})"

class Tan:
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
        return torch.tan(agraph_evals[node1])

    def _string(agraph_strings, node1, node2):
        """
        Parameters
        ----------
        agraph_strings : [Argument]
        node1 : [Argument]
        node2 : [Argument]

        """
        return f"tan({agraph_strings[node1]})"

class ASin:

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
        return torch.asin(agraph_evals[node1])

    def _string(agraph_strings, node1, node2):
        """
        Parameters
        ----------
        agraph_strings : [Argument]
        node1 : [Argument]
        node2 : [Argument]

        """
        return f"asin({agraph_strings[node1]})"

class ACos:

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
        return torch.acos(agraph_evals[node1])

    def _string(agraph_strings, node1, node2):
        """
        Parameters
        ----------
        agraph_strings : [Argument]
        node1 : [Argument]
        node2 : [Argument]

        """
        return f"acos({agraph_strings[node1]})"

class ATan:

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
        return torch.atan(agraph_evals[node1])

    def _string(agraph_strings, node1, node2):
        """
        Parameters
        ----------
        agraph_strings : [Argument]
        node1 : [Argument]
        node2 : [Argument]

        """
        return f"atan({agraph_strings[node1]})"

class Sinh:

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
        return torch.sinh(agraph_evals[node1])

    def _string(agraph_strings, node1, node2):
        """
        Parameters
        ----------
        agraph_strings : [Argument]
        node1 : [Argument]
        node2 : [Argument]

        """
        return f"sinh({agraph_strings[node1]})"

class Cosh:

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
        return torch.cosh(agraph_evals[node1])

    def _string(agraph_strings, node1, node2):
        """
        Parameters
        ----------
        agraph_strings : [Argument]
        node1 : [Argument]
        node2 : [Argument]

        """
        return f"cosh({agraph_strings[node1]})"

class Tanh:

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
        return torch.tanh(agraph_evals[node1])

    def _string(agraph_strings, node1, node2):
        """
        Parameters
        ----------
        agraph_strings : [Argument]
        node1 : [Argument]
        node2 : [Argument]

        """
        return f"tanh({agraph_strings[node1]})"
