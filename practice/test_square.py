import numpy as np
import torch
import sys;sys.path.append("../")
from genepy.equation import Equation


if __name__ == "__main__":
    expression = "X_0 ** 2" 
    equation = Equation(expression=expression)
    X = np.hstack(
        (
            np.linspace(0, np.pi, 100).reshape((-1, 1)),
            np.linspace(np.pi, 2 * np.pi, 100).reshape((-1, 1)),
        )
    )
    _y = np.sin(X[:,1]) + np.sin(np.sin(X[:,1]))/3 + 4 + np.log(X[:,1]*X[:,0])
    _y = _y.reshape((-1,1))
    X_torch = torch.from_numpy(X)
    X_torch.requires_grad = True
    y = equation.evaluate_equation(X_torch)
    import pdb;pdb.set_trace()
