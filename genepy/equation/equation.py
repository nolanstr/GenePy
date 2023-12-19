import warnings
import numpy as np
import torch
from torch.autograd import grad
import re
pattern = re.compile(r'C_\d+')

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
        elif (genotype is None) and (expression is not None):
            self._init_from_expression(expression)
        else:
            warnings.warn(
                "Specrified both genotype and expression, initializing from expression"
            )
            self._init_from_expression(expression)

    def _init_from_genotype(self, genotype):
        self._genotype = genotype
        self._expression = expression_from_genotype(genotype)
        self._get_simplified_genotype()
        self._set_number_of_constants()
        self._expression = expression_from_genotype(self._simplified_genotype)

    def _init_from_expression(self, expression):
        structure_dict, ints, floats, genotype = genotype_from_expression(expression)
        self._genotype = genotype
        self._simplified_genotype = genotype
        self._set_number_of_constants()
        self._expression = expression_from_genotype(genotype)
    
    def _set_number_of_constants(self):
        self.number_of_iconstants = np.sum(self._simplified_genotype[:, 0] == -1)
        self.number_of_constants = np.sum(self._simplified_genotype[:, 0] == 0)
        self.constants = torch.ones((1, self.number_of_constants))

    def _get_simplified_genotype(self):
        _, _, _, genotype = genotype_from_expression(self._expression)
        self._simplified_genotype = genotype

    def set_constants(self, constants):
        if isinstance(constants, np.ndarray):
            constants = torch.from_numpy(constants)
            constants.requires_grad = True
        if constants.dim() == 1:
            constants = constants.reshape((1,-1))
        self.constants = constants
        self._expression = expression_from_genotype(self.genotype)
        self._set_constants_in_expression()

    def _set_constants_in_expression(self):
        C_values = list(set(pattern.findall(self._expression))) 
        tmp_string = self._expression
        for i, C_i in enumerate(C_values):
            tmp_string = tmp_string.replace(C_i, str(round(self.constants[0, i].item(), 3)))
        self._expression = tmp_string

    def evaluate_equation(self, X):
        equation_output = forward_eval(self._simplified_genotype, 
                                                self.constants, X)
        return equation_output
    
    
    def evaluate_equation_derivative_wrt_x(self, X):
        f = self.evaluate_equation(X)
        df_dx = torch.hstack(grad(f.sum(), X, retain_graph=True,
                                            materialize_grads=True))
        return f, df_dx

    def evaluate_equation_derivative_wrt_c(self, X):
        f = self.evaluate_equation(X)
        df_dc = self._compute_derivative_wrt_c(f) 
        return f, df_dc

    def _compute_derivative_wrt_c(self, f):
        df_dc = torch.zeros((f.shape[0], f.shape[1], self.constants.shape[1]))
        try:
            for i, f_i in enumerate(f):
                for j, f_j in enumerate(f_i):
                    df_dc[i,j,:] = grad(f_j.sum(), self.constants,
                            retain_graph=True, materialize_grads=True)[0]
        except:
            import pdb;pdb.set_trace()
        return df_dc
    
    def copy(self):
        equation = Equation(genotype=self.genotype)
        equation.fitness = self.fitness
        return equation
