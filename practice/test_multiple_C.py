import torch
import numpy as np

from genepy.nodes.base_nodes import Var, Const, Add, Sub, Mult, Div

NODE_DICT = {"0": Const, "1": Var, "2": Add, "3": Sub, "4": Mult, "5": Div}


def forward_eval(operator_id, _x, agraph_evals, agraph_consts, node1, node2):
    op = NODE_DICT[str(operator_id)]
    return op.forward(_x, agraph_evals, agraph_consts, node1, node2)


def backward_eval(y, _x, ders):
    """
    dy_dx shape --> (nDers, nP, nX, dX)
    """
    #_x = torch.tile(_x, (nP, 1, 1))
    dy_dx = [perform_grad(y, _x)]

    for _ in range(1, ders):
        dy_dx.append(perform_grad(dy_dx[-1], _x))
    dy_dx = torch.stack(dy_dx)

    return dy_dx


def perform_grad(y, _x):
    nP, nX, dX = y.shape
    dy_dx =  list(torch.autograd.grad(y, [_x]*nP, 
            grad_outputs=[torch.ones_like(y)/nP]*nP, create_graph=True,
            allow_unused=True))
    for i in range(nP):
        if dy_dx[i] is None:
            dy_dx[i] = torch.zeros_like(_x, requires_grad=True)
    dy_dx = torch.stack(dy_dx)
    return dy_dx


def forward_string(operator_id, agraph_strings, node1, node2):
    op = NODE_DICT[str(operator_id)]
    return op._string(agraph_strings, node1, node2)


agraph_command_array = np.array(
    [[1, 0, 0], [0, 0, 0], [0, 1, 1], [0, 2, 2], [2, 0, 1], [4, 4, 0], [3, 5, 3]]
)
_X = torch.arange(0, 5, dtype=torch.float64).reshape((-1, 1))
_X.requires_grad = True
agraph_consts = torch.tensor([2.0, 3.0, 4.0], requires_grad=True).repeat(4, 1)

agraph_evals = []
agraph_strings = []

for row in agraph_command_array:
    agraph_evals.append(
        forward_eval(row[0], _X, agraph_evals, agraph_consts, row[1], row[2])
    )
    agraph_strings.append(forward_string(row[0], agraph_strings, row[1], row[2]))
model_eval = agraph_evals[-1]
model_string = agraph_strings[-1]
dy_dx = backward_eval(model_eval, _X, 2)

print(model_string)
import pdb;pdb.set_trace()
