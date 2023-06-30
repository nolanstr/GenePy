import torch
import numpy as np
import re
import sympy
import sympy.parsing as sp
from sympy import symbols, srepr, sympify
from sympy.core.traversal import postorder_traversal, preorder_traversal
from sympy.core.numbers import Float, Integer, Rational

op_dict = {"pconst":-1,
           "cconst":0,
           "var":1,
           "add":2,
           "sub":3,
           "mul":4,
           "div":5,
           "pow":6,
           "sqrt":7,
           "exp":8,
           "log":9,
           "sin":10,
           "cos":11,
           "tan":12,
           "asin":13,
           "acos":14,
           "atan":15,
           "sinh":16,
           "cosh":17,
           "tanh":18}
"""
Curently, sympy does not support a sympy.Sub expression and thus all expressions
with subtraction are replace with addition and multiplication. I.e. "1-2X" gives
sympy.Add(1, symp.Mul(2,X)).
This is consistent with divide (replaced by multiply) and sqrt (replaced by
power).
"""

def genotype_from_phenotype(structure, simplify=True):
    expression = make_sympy_expression(structure)
    if simplify:
        ints, floats = extract_constants(expression)
        expression = replace_constants(expression, ints)
        expression = sympify(expression, evaluate=True)
        ints, floats = extract_constants(expression)
        expression = replace_constants(expression, floats)
    else:
        ints, floats = extract_constants(expression)
        expression = replace_constants(expression, ints)
        expression = replace_constants(expression, floats)

    row = 0
    v_count = 0
    c_count = 0
    genes = []
    def traverse(node):
        nonlocal row, v_count, c_count, genes

        if isinstance(node, sympy.Symbol):
            if "X" in str(node):
                genes.append([[1, v_count, v_count], row])
                v_count += 1
                row += 1
                return {"var": {"symbol":str(node), "gene":genes[-1]}}
            elif "C" in str(node):
                genes.append([[0, c_count, c_count], row])
                c_count += 1
                row += 1
                return {"const": {"symbol":str(node), "gene":genes[-1]}}
            else:
                genes.append([[-1, eval(str(node)[2:]), eval(str(node)[2:])], row])
                row += 1
                return {"pconst": {"symbol":str(node), "gene":genes[-1]}}

        else:
            op = get_op(node)
            args = node.args
            sub_structure = []
            for arg in args:
                sub_structure.append(traverse(arg))
            return nest_structure(sub_structure, "mul")

    def nest_structure(structure, op):
        nonlocal row, v_count, c_count, genes
        if len(structure) == 1:
            return structure[0]
        else:
            row1 = structure[0][next(iter(structure[0]))]["gene"][1]
            row2 = structure[1][next(iter(structure[1]))]["gene"][1]
            genes.append([[op_dict[op], row1, row2], row])
            row += 1
            return {op: {"symbol":[structure[0], nest_structure(structure[1:],
                op)], "gene":genes[-1]}}

    structure_dict = traverse(expression)
    genes_order = [gene[1] for gene in genes]
    genes = np.vstack([np.array(gene[0], dtype=object) for gene in genes])[np.argsort(genes_order),:]

    return structure_dict, ints, floats, genes

def make_sympy_expression(string):
    if isinstance(string, sympy.Expr):
        return string
    pattern = r"[A-Za-z_]\w*"
    variables = list(set(re.findall(pattern, string)))
    symbols = [sympy.Symbol(v) for v in variables]
    for var in variables:
        i = var[2:]
        string = string.replace(f"X_{i}", f"symbols[{i}]")
    return eval(string)

def extract_constants(expression):
    ints = {}
    floats = {}
    for term in expression.atoms():
        if term.is_constant():
            if isinstance(term, Integer):
                ints[f"I_{term}"] = int(term)
            elif isinstance(term, Float):
                floats[f"C_{len(floats)}"] = float(term)
            elif isinstance(term, Rational):
                ints[f"R_{term}"] = term
            else:
                raise NotImplementedError("extract constant unknown")
    return ints, floats


def replace_constants(expression, consts_dict):
    replacement_symbols = [sympy.Symbol(symbol) for symbol in consts_dict.keys()]
    replaced_expression = expression.subs(
        list(zip(consts_dict.values(), replacement_symbols))
    )
    return replaced_expression


def replace_symbols(expression, consts_dict):
    replacement_symbols = [sympy.Symbol(symbol) for symbol in consts_dict.keys()]
    replacement_values = [
        sympy.UnevaluatedExpr(symbol) for symbol in consts_dict.values()
    ]
    replaced_expression = expression.subs(
        list(zip(replacement_symbols, replacement_values)), evaluate=False
    )
    return replaced_expression

def get_op(node):
    if isinstance(node, sympy.Add):
        return "add"
    #elif isinstance(node, sympy.Sub):
    #    return "sub"
    elif isinstance(node, sympy.Mul):
        return "mul"
    #elif isinstance(node, sympy.Div):
    #    return "div"
    elif isinstance(node, sympy.Pow):
        return "pow"
    #elif isinstance(node, sympy.sqrt):
    #    return "sqrt"
    elif isinstance(node, sympy.exp):
        return "exp"
    elif isinstance(node, sympy.log):
        return "log"
    elif isinstance(node, sympy.sin):
        return "sin"
    elif isinstance(node, sympy.cos):
        return "cos"
    elif isinstance(node, sympy.tan):
        return "tan"
    elif isinstance(node, sympy.asin):
        return "asin"
    elif isinstance(node, sympy.acos):
        return "acos"
    elif isinstance(node, sympy.atan):
        return "atan"
    else:
        raise NotImplementedError(
        "operator not implemented/bad phenotype description")
