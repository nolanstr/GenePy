import torch


class Var:
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
        with torch.enable_grad():
            _var = _x[None, :, node1, None]
        return _var

    def _string(genotype_strings, node1, node2):
        """
        Parameters
        ----------
        genotype_strings : [Argument]
        node1 : [Argument]
        node2 : [Argument]

        """
        return f"X_{node1}"


class Const:
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
        with torch.enable_grad():
            _consts = genotype_consts[:, node1, None, None]
        return _consts

    def _string(genotype_strings, node1, node2):
        """
        Parameters
        ----------
        genotype_strings : [Argument]
        node1 : [Argument]
        node2 : [Argument]

        """
        return f"C_{node1}"


class IConst:
    def forward(_x, genotype_evals, genotype_consts, genotype_Iconsts, node1, node2):
        """
        Changed from true as constants cant have requires grad

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
        try:
            _Iconsts = genotype_Iconsts[:, node1, None, None]
        except:
            import pdb

            pdb.set_trace()
        return _Iconsts

    def _string(genotype_strings, node1, node2):
        """
        Parameters
        ----------
        genotype_strings : [Argument]
        node1 : [Argument]
        node2 : [Argument]

        """
        return f"I_{node1}"


class Add:
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
        return torch.add(genotype_evals[node1], genotype_evals[node2])

    def _string(genotype_strings, node1, node2):
        """
        Parameters
        ----------
        genotype_strings : [Argument]
        node1 : [Argument]
        node2 : [Argument]

        """
        return f"({genotype_strings[node1]} + {genotype_strings[node2]})"


class Sub:
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
        return torch.subtract(genotype_evals[node1], genotype_evals[node2])

    def _string(genotype_strings, node1, node2):
        """
        Parameters
        ----------
        genotype_strings : [Argument]
        node1 : [Argument]
        node2 : [Argument]

        """
        return f"({genotype_strings[node1]} - {genotype_strings[node2]})"


class Mult:
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
        return torch.multiply(genotype_evals[node1], genotype_evals[node2])

    def _string(genotype_strings, node1, node2):
        """
        Parameters
        ----------
        genotype_strings : [Argument]
        node1 : [Argument]
        node2 : [Argument]

        """
        return f"({genotype_strings[node1]} * {genotype_strings[node2]})"


class Div:
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
        return torch.divide(genotype_evals[node1], genotype_evals[node2])

    def _string(genotype_strings, node1, node2):
        """
        Parameters
        ----------
        genotype_strings : [Argument]
        node1 : [Argument]
        node2 : [Argument]

        """
        return f"({genotype_strings[node1]} * {genotype_strings[node2]})"


class Pow:
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
        return torch.pow(genotype_evals[node1], genotype_evals[node2])

    def _string(genotype_strings, node1, node2):
        """
        Parameters
        ----------
        genotype_strings : [Argument]
        node1 : [Argument]
        node2 : [Argument]

        """
        return f"({genotype_strings[node1]})^({genotype_strings[node2]})"


class Square:
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
        return torch.square(genotype_evals[node1])

    def _string(genotype_strings, node1, node2):
        """
        Parameters
        ----------
        genotype_strings : [Argument]
        node1 : [Argument]
        node2 : [Argument]

        """
        return f"({genotype_strings[node1]})^(2)"


class Sqrt:
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
        return torch.sqrt(genotype_evals[node1])

    def _string(genotype_strings, node1, node2):
        """
        Parameters
        ----------
        genotype_strings : [Argument]
        node1 : [Argument]
        node2 : [Argument]

        """
        return f"sqrt({genotype_strings[node1]})"


class Exp:
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
        return torch.exp(genotype_evals[node1])

    def _string(genotype_strings, node1, node2):
        """
        Parameters
        ----------
        genotype_strings : [Argument]
        node1 : [Argument]
        node2 : [Argument]

        """
        return f"exp({genotype_strings[node1]})"


class Log:
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
        return torch.log(genotype_evals[node1])

    def _string(genotype_strings, node1, node2):
        """
        Parameters
        ----------
        genotype_strings : [Argument]
        node1 : [Argument]
        node2 : [Argument]

        """
        return f"log({genotype_strings[node1]})"
