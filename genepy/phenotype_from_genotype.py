from genepy.nodes.base_nodes import *
from genepy.nodes.trig_nodes import *

NODE_DICT = {"-1":IConst, 
             "0": Const,
             "1": Var,
             "2": Add,
             "3": Sub,
             "4": Mult,
             "5": Div,
             "6":Pow,
             "7":Sqrt,
             "8":Exp,
             "9":Log,
             "10":Sin,
             "11":Cos,
             "12":Tan,
             "13":ASin,
             "14":ACos,
             "15":ATan,
             "16":Sinh,
             "17":Cosh,
             "18":Tanh
}

def phenotype_from_genotype(genotype):
    agraph_strings = []
    for row in genotype:
        agraph_strings.append(forward_string(row[0], agraph_strings, row[1], row[2]))

    model_string = agraph_strings[-1]
    return model_string

def forward_string(operator_id, agraph_strings, node1, node2):
    op = NODE_DICT[str(operator_id)]
    return op._string(agraph_strings, node1, node2)