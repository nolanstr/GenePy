import sympy as s
from sympy import srepr
from sympy.core.traversal import postorder_traversal, preorder_traversal
import sympy.parsing as sp

variables = s.symbols("X_0")
eq = "X_0 + 2."
sp_eq = sp.parse_expr(eq)
sr_eq = srepr(sp_eq)

for arg in postorder_traversal(sp_eq):
    print(arg)

for arg in preorder_traversal(sp_eq):
    print(arg)
import pdb;pdb.set_trace()
