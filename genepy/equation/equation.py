import warnings
import numpy as np
import torch
from torch.autograd import grad
import re

C_pattern = re.compile(r"C_\d+")
I_pattern = re.compile(r"I_\d+")

from ..organism import Organism
from .genotype_from_expression import genotype_from_expression
from .expression_from_genotype import expression_from_genotype
from .equation_traversal import forward_eval


class Equation(Organism):
    def __init__(self, genotype=None, expression=None):
        """
        Parameters
        ----------
        self : object [Argument]
        genotype :default: None [Argument]
        expression :default: None [Argument]

        """
        super().__init__()
        self._set_genetic_information(genotype, expression)

    def __str__(self):
        """
        Parameters
        ----------
        self : object [Argument]

        """
        return self.expression

    def _set_genetic_information(self, genotype, expression):
        """
        Parameters
        ----------
        self : object [Argument]
        genotype : [Argument]
        expression : [Argument]

        """
        if (genotype is not None) and (expression is None):
            self._init_from_genotype(genotype)
        elif (genotype is None) and (expression is not None):
            self._init_from_expression(expression)
        else:
            warnings.warn(
                "Specrified both genotype and expression, initializing from expression"
            )
            self._init_from_expression(expression)

    def _init_from_genotype(self, genotype):
        """
        Parameters
        ----------
        self : object [Argument]
        genotype : [Argument]

        """
        self._genotype = genotype
        self.expression = expression_from_genotype(genotype)
        self._stored_expression = self.expression
        self._get_simplified_genotype()
        self.expression = expression_from_genotype(self._simplified_genotype)

    def _init_from_expression(self, expression):
        """
        Parameters
        ----------
        self : object [Argument]
        expression : [Argument]

        """
        structure_dict, ints, floats, genotype = genotype_from_expression(expression)
        self._genotype = genotype
        self._simplified_genotype = genotype
        self._set_number_of_constants(ints, floats)
        self.expression = expression_from_genotype(genotype)

    def _set_number_of_constants(self, ints, floats):
        """
        Parameters
        ----------
        self : object [Argument]
        ints : [Argument]
        floats : [Argument]

        """
        self.number_of_iconstants = len(ints)
        self.number_of_constants = len(floats)
        self.Iconstants = torch.tensor(
            [float(val) for val in list(ints.values())]
        ).reshape((1, -1))
        self.constants = torch.tensor(list(floats.values())).reshape((1, -1))
        self.constants.requires_grad = True
        self.Iconstants.requires_grad = True

    def _get_simplified_genotype(self):
        """
        Parameters
        ----------
        self : object [Argument]

        """
        _, ints, floats, genotype = genotype_from_expression(self.expression)
        self._set_number_of_constants(ints, floats)
        self._simplified_genotype = genotype

    def set_constants(self, constants):
        """
        Parameters
        ----------
        self : object [Argument]
        constants : [Argument]

        """
        if isinstance(constants, np.ndarray):
            constants = torch.from_numpy(constants)
            constants.requires_grad = True
        if constants.dim() == 1:
            constants = constants.reshape((1, -1))
        self.constants = constants
        self.expression = expression_from_genotype(self._simplified_genotype)
        self._set_constants_in_expression()

    def _set_constants_in_expression(self):
        """
        Parameters
        ----------
        self : object [Argument]

        """
        C_values = list(set(C_pattern.findall(self.expression)))
        I_values = list(set(I_pattern.findall(self.expression)))
        tmp_string = self.expression
        for i, C_i in enumerate(C_values):
            try:
                tmp_string = tmp_string.replace(
                    C_i, str(round(self.constants[0, i].item(), 3))
                )
            except:
                import pdb

                pdb.set_trace()
        for i, I_i in enumerate(I_values):
            tmp_string = tmp_string.replace(
                I_i, str(round(self.Iconstants[0, i].item(), 3))
            )
        self.expression = tmp_string

    def evaluate_equation(self, X):
        """
        Parameters
        ----------
        self : object [Argument]
        X : [Argument]

        """
        equation_output = forward_eval(
            self._simplified_genotype, self.constants, self.Iconstants, X
        )
        return equation_output

    def evaluate_equation_derivative_wrt_x(self, X):
        """
        Parameters
        ----------
        self : object [Argument]
        X : [Argument]

        """
        f = self.evaluate_equation(X)
        df_dx = torch.hstack(
            grad(f.sum(), X, retain_graph=True, materialize_grads=True)
        )
        return f, df_dx

    def evaluate_equation_derivative_wrt_c(self, X):
        """
        Parameters
        ----------
        self : object [Argument]
        X : [Argument]

        """
        f = self.evaluate_equation(X)
        df_dc = self._compute_derivative_wrt_c(f)
        return f, df_dc

    def _compute_derivative_wrt_c(self, f):
        """
        Parameters
        ----------
        self : object [Argument]
        f : [Argument]

        """
        df_dc = torch.zeros((f.shape[0], f.shape[1], self.constants.shape[1]))
        for i, f_i in enumerate(f):
            for j, f_j in enumerate(f_i):
                df_dc[i, j, :] = grad(
                    f_j.sum(), self.constants, retain_graph=True, materialize_grads=True
                )[0]
        return df_dc

    def copy(self):
        """
        Parameters
        ----------
        self : object [Argument]

        """
        equation = Equation(genotype=self.genotype)
        equation.fitness = self.fitness
        equation.set_constants(self.constants)

        return equation
