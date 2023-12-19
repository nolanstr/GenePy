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


def expression_from_genotype(genotype):
    """
    Parameters
    ----------
    genotype : [Argument]

    """
    agraph_strings = []
    for row in genotype:
        agraph_strings.append(forward_string(row[0], agraph_strings, row[1], row[2]))
    model_string = agraph_strings[-1]
    return model_string


def forward_string(operator_id, agraph_strings, node1, node2):
    """
    Parameters
    ----------
    operator_id : [Argument]
    agraph_strings : [Argument]
    node1 : [Argument]
    node2 : [Argument]

    """
    op = NODE_DICT[str(operator_id)]
    return op._string(agraph_strings, node1, node2)
