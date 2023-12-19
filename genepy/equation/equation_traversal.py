import numpy as np
import torch

from genepy.nodes.base_nodes import *
from genepy.nodes.trig_nodes import *

NODE_DICT = {
    "-1": IConst,
    "0": Const,
    "1": Var,
    "2": Add,
    "3": Sub,
    "4": Mult,
    "5": Div,
    "6": Pow,
    "7": Square,
    "8": Sqrt,
    "9": Exp,
    "10": Log,
    "11": Sin,
    "12": Cos,
    "13": Tan,
    "14": ASin,
    "15": ACos,
    "16": ATan,
    "17": Sinh,
    "18": Cosh,
    "19": Tanh,
}

leaf_nodes = [-1, 0, 1]
univariate_nodes = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
bivariate_nodes = [2, 3, 4, 5, 6]


def forward_eval(genotype, constants, X):
    """
    Parameters
    ----------
    genotype : [Argument]
    constants : [Argument]
    X : [Argument]

    """
    genotype_evals = []
    for gene in genotype:
        try:
            genotype_evals.append(
                _forward_eval(gene[0], X, genotype_evals, constants, gene[1], gene[2])
            )
        except:
            print(genotype, constants)
    equation_forward_eval = genotype_evals[-1]
    return equation_forward_eval


def _forward_eval(operator_id, _x, genotype_evals, genotype_consts, node1, node2):
    """
    Parameters
    ----------
    operator_id : [Argument]
    _x : [Argument]
    genotype_evals : [Argument]
    genotype_consts : [Argument]
    node1 : [Argument]
    node2 : [Argument]

    """
    op = NODE_DICT[str(operator_id)]
    return op.forward(_x, genotype_evals, genotype_consts, node1, node2)
