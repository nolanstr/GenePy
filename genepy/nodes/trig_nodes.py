import torch


class Sin:
    def forward(_x, genotype_evals, genotype_consts, genotype_Iconsts, node1, node2):
        """
        Parameters
        ----------
        _x : [Argument]
        genotype_evals : [Argument]
        genotype_consts : [Argument]
            node1 : [Argument]
        node2 : [Argument]
        genotype_Iconsts : [Argument]
        node1 : [Argument]
        node2 : [Argument]

        """
        return torch.sin(genotype_evals[node1])

    def _string(genotype_strings, node1, node2):
        """
        Parameters
        ----------
        genotype_strings : [Argument]
        node1 : [Argument]
        node2 : [Argument]

        """
        return f"sin({genotype_strings[node1]})"


class Cos:
    def forward(_x, genotype_evals, genotype_consts, genotype_Iconsts, node1, node2):
        """
        Parameters
        ----------
        _x : [Argument]
        genotype_evals : [Argument]
        genotype_consts : [Argument]
            node1 : [Argument]
        node2 : [Argument]
        genotype_Iconsts : [Argument]
        node1 : [Argument]
        node2 : [Argument]

        """
        return torch.cos(genotype_evals[node1])

    def _string(genotype_strings, node1, node2):
        """
        Parameters
        ----------
        genotype_strings : [Argument]
        node1 : [Argument]
        node2 : [Argument]

        """
        return f"cos({genotype_strings[node1]})"


class Tan:
    def forward(_x, genotype_evals, genotype_consts, genotype_Iconsts, node1, node2):
        """
        Parameters
        ----------
        _x : [Argument]
        genotype_evals : [Argument]
        genotype_consts : [Argument]
            node1 : [Argument]
        node2 : [Argument]
        genotype_Iconsts : [Argument]
        node1 : [Argument]
        node2 : [Argument]

        """
        return torch.tan(genotype_evals[node1])

    def _string(genotype_strings, node1, node2):
        """
        Parameters
        ----------
        genotype_strings : [Argument]
        node1 : [Argument]
        node2 : [Argument]

        """
        return f"tan({genotype_strings[node1]})"


class ASin:
    def forward(_x, genotype_evals, genotype_consts, genotype_Iconsts, node1, node2):
        """
        Parameters
        ----------
        _x : [Argument]
        genotype_evals : [Argument]
        genotype_consts : [Argument]
            node1 : [Argument]
        node2 : [Argument]
        genotype_Iconsts : [Argument]
        node1 : [Argument]
        node2 : [Argument]

        """
        return torch.asin(genotype_evals[node1])

    def _string(genotype_strings, node1, node2):
        """
        Parameters
        ----------
        genotype_strings : [Argument]
        node1 : [Argument]
        node2 : [Argument]

        """
        return f"asin({genotype_strings[node1]})"


class ACos:
    def forward(_x, genotype_evals, genotype_consts, genotype_Iconsts, node1, node2):
        """
        Parameters
        ----------
        _x : [Argument]
        genotype_evals : [Argument]
        genotype_consts : [Argument]
            node1 : [Argument]
        node2 : [Argument]
        genotype_Iconsts : [Argument]
        node1 : [Argument]
        node2 : [Argument]

        """
        return torch.acos(genotype_evals[node1])

    def _string(genotype_strings, node1, node2):
        """
        Parameters
        ----------
        genotype_strings : [Argument]
        node1 : [Argument]
        node2 : [Argument]

        """
        return f"acos({genotype_strings[node1]})"


class ATan:
    def forward(_x, genotype_evals, genotype_consts, genotype_Iconsts, node1, node2):
        """
        Parameters
        ----------
        _x : [Argument]
        genotype_evals : [Argument]
        genotype_consts : [Argument]
            node1 : [Argument]
        node2 : [Argument]
        genotype_Iconsts : [Argument]
        node1 : [Argument]
        node2 : [Argument]

        """
        return torch.atan(genotype_evals[node1])

    def _string(genotype_strings, node1, node2):
        """
        Parameters
        ----------
        genotype_strings : [Argument]
        node1 : [Argument]
        node2 : [Argument]

        """
        return f"atan({genotype_strings[node1]})"


class Sinh:
    def forward(_x, genotype_evals, genotype_consts, genotype_Iconsts, node1, node2):
        """
        Parameters
        ----------
        _x : [Argument]
        genotype_evals : [Argument]
        genotype_consts : [Argument]
            node1 : [Argument]
        node2 : [Argument]
        genotype_Iconsts : [Argument]
        node1 : [Argument]
        node2 : [Argument]

        """
        return torch.sinh(genotype_evals[node1])

    def _string(genotype_strings, node1, node2):
        """
        Parameters
        ----------
        genotype_strings : [Argument]
        node1 : [Argument]
        node2 : [Argument]

        """
        return f"sinh({genotype_strings[node1]})"


class Cosh:
    def forward(_x, genotype_evals, genotype_consts, genotype_Iconsts, node1, node2):
        """
        Parameters
        ----------
        _x : [Argument]
        genotype_evals : [Argument]
        genotype_consts : [Argument]
            node1 : [Argument]
        node2 : [Argument]
        genotype_Iconsts : [Argument]
        node1 : [Argument]
        node2 : [Argument]

        """
        return torch.cosh(genotype_evals[node1])

    def _string(genotype_strings, node1, node2):
        """
        Parameters
        ----------
        genotype_strings : [Argument]
        node1 : [Argument]
        node2 : [Argument]

        """
        return f"cosh({genotype_strings[node1]})"


class Tanh:
    def forward(_x, genotype_evals, genotype_consts, genotype_Iconsts, node1, node2):
        """
        Parameters
        ----------
        _x : [Argument]
        genotype_evals : [Argument]
        genotype_consts : [Argument]
            node1 : [Argument]
        node2 : [Argument]
        genotype_Iconsts : [Argument]
        node1 : [Argument]
        node2 : [Argument]

        """
        return torch.tanh(genotype_evals[node1])

    def _string(genotype_strings, node1, node2):
        """
        Parameters
        ----------
        genotype_strings : [Argument]
        node1 : [Argument]
        node2 : [Argument]

        """
        return f"tanh({genotype_strings[node1]})"
