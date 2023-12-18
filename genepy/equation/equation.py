import warnings
import numpy as np
import torch

from ..organism import Organism
from .genotype_from_expression import genotype_from_expression
from .expression_from_genotype import expression_from_genotype
from .equation_traversal import forward_eval


class Equation(Organism):
    def __init__(self, genotype=None, expression=None):
        super().__init__()
        self._set_genetic_information(genotype, expression)
    
    def __str__(self):
        return self._expression

    def _set_genetic_information(self, genotype, expression):
        if (genotype is not None) and (expression is None):
            self._init_from_genotype(genotype)
            self._get_simplified_genotype()
        elif (genotype is None) and (expression is not None):
            self._init_from_expression(expression)
        else:
            warnings.warn(
                "Specrified both genotype and expression, initializing from expression"
            )
            self._init_from_expression(expression)

    def _init_from_genotype(self, genotype):
        self._genotype = genotype
        self._set_number_of_constants()
        self._expression = expression_from_genotype(genotype)

    def _init_from_expression(self, expression):
        structure_dict, ints, floats, genotype = genotype_from_expression(expression)
        self._genotype = genotype
        self._set_number_of_constants()
        self._simplified_genotype = genotype
        self._expression = expression_from_genotype(genotype)

    def _set_number_of_constants(self):
        self.number_of_constants = np.sum(self.genotype[:, 0] == 1)
        self.number_of_iconstants = np.sum(self.genotype[:, 0] == -1)
        self.constants = torch.ones((1, self.number_of_constants))

    def _get_simplified_genotype(self):
        _, _, _, genotype = genotype_from_expression(self._expression)
        self._simplified_genotype = genotype

    def evaluate_equation(self, X):
        equation_output = forward_eval(self._simplified_genotype, self.constants, X)
        return equation_output
