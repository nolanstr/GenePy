import torch
import numpy as np

from genepy.nodes.base_nodes import Var, Const, Add, Sub, Mult, Div

NODE_DICT = {"0": Const, "1": Var, "2": Add, "3": Sub, "4": Mult, "5": Div}


def forward_eval(operator_id, _x, agraph_evals, agraph_consts, node1, node2):
    op = NODE_DICT[str(operator_id)]
    return op.forward(_x, agraph_evals, agraph_consts, node1, node2)


def backward_eval(y, _x, ders):
    dy_dx = [perform_grad(y.repeat(1,_x.shape[1]), _x)]
    for _ in range(1, ders):
        dy_dx.append(perform_grad(dy_dx[-1], _x))

    dy_dx = torch.stack(dy_dx)
    return dy_dx


def perform_grad(y, _x):
    dy_dx = torch.autograd.grad(
        y, _x, grad_outputs=torch.ones_like(_x), create_graph=True, allow_unused=True
    )[0]
    if dy_dx is None:
        dy_dx = torch.zeros_like(_x, requires_grad=True)
    return dy_dx


def forward_string(operator_id, agraph_strings, node1, node2):
    op = NODE_DICT[str(operator_id)]
    return op._string(agraph_strings, node1, node2)


agraph_command_array = np.array(
    [[1, 0, 0], [1, 1, 1], [0, 0, 0], [0, 1, 1], [0, 2, 2], [2, 0, 2], [2, 5,
        1],  [4, 6, 3], [3, 7, 4]]
)
# (((X_0 + X_1 + C_0) * C_1) - C_2)
_X = torch.tensor(np.hstack([np.arange(0, 6).reshape((-1,1)),
                             np.arange(0,6).reshape((-1,1))]),
        dtype=torch.float64, requires_grad=True)
agraph_consts = torch.tensor([[2.0, 3.0, 4.0]], requires_grad=True)
agraph_evals = []
agraph_strings = []

for row in agraph_command_array:
    agraph_evals.append(
        forward_eval(row[0], _X, agraph_evals, agraph_consts, row[1], row[2])
    )
    agraph_strings.append(forward_string(row[0], agraph_strings, row[1], row[2]))
model_eval = agraph_evals[-1]
model_string = agraph_strings[-1]
print(model_string)
dy_dx = backward_eval(model_eval, _X, 2)

_Y = ((_X + agraph_consts[0, 0]) * agraph_consts[0, 1]) - agraph_consts[0, 2]

import pdb;pdb.set_trace()
