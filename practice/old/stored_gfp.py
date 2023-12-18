import torch
import numpy as np
import sympy
import sympy.parsing as sp
from sympy import symbols, srepr, sympify
from sympy.core.traversal import postorder_traversal, preorder_traversal
from sympy.core.numbers import Float, Integer

op_dict = {"pconst":-1,
           "cconst":0,
           "var":1,
           "add":2,
           "sub":3,
           "mul":4,
           "div":5,
           "pow":6,
           "sin":7,
           "cos":8,
           "tan":9,
           "sin":10,
           "cos":11,
           "tan":12}

def extract_constants(expression):
    ints = {}
    floats = {}
    for term in expression.atoms():
        if term.is_constant():
            if isinstance(term, Integer):
                ints[f"I_{term}"] = int(term)
            else:
                floats[f"C_{len(floats)}"] = float(term)
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



def genetype_from_string(structure, simplify=False):
    expression = sympify(structure, evaluate=False)
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
                genes.append([[-1, int(str(node)[2:]), int(str(node)[2:])], row])
                row += 1
                return {"pconst": {"symbol":str(node), "gene":genes[-1]}}

        elif isinstance(node, sympy.Mul):
            args = node.args
            sub_structure = []

            for arg in args:
                sub_structure.append(traverse(arg))
            return nest_structure(sub_structure, "mul")

        elif isinstance(node, sympy.Add):
            args = node.args
            sub_structure = []
            for arg in args:
                sub_structure.append(traverse(arg))
            return nest_structure(sub_structure, "add")

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
    genes = np.vstack([gene[0] for gene in genes])[np.argsort(genes_order),:]
    return structure_dict, ints, floats, genes

