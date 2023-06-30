import torch

import sympy
import sympy.parsing as sp
from sympy import symbols, srepr, sympify
from sympy.core.traversal import postorder_traversal, preorder_traversal
from sympy.core.numbers import Float, Integer

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
    replaced_expression = expression.subs(list(zip(consts_dict.values(), replacement_symbols)))
    return replaced_expression

def replace_symbols(expression, consts_dict):
    replacement_symbols = [sympy.Symbol(symbol) for symbol in consts_dict.keys()]
    replacement_values = [sympy.UnevaluatedExpr(symbol) for symbol in consts_dict.values()]
    replaced_expression = expression.subs(list(zip(replacement_symbols,
                                replacement_values)), evaluate=False)
    return replaced_expression

def nest_structure(structure, op):

    if len(structure) == 1:
        return structure[0]
    else:
        return {op: [structure[0], nest_structure(structure[1:], op)]}

def parse_expression_structure(structure, simplify=False):
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

    def traverse(node):
        row = 0
        v_count = 0
        c_count = 0
        genes = []

        if isinstance(node, sympy.Symbol):
            if "X" in str(node):
                genes.append([[1, v_count, v_count], row])
                v_count += 1
                row += 1
                return {"var": str(node)}
            elif "C" in str(node):
                genes.append([[0, c_count, c_count], row])
                c_count += 1
                row += 1
                return {"Const": str(node)}
            else:
                import pdb;pdb.set_trace()
                genes.append([[-1, int(str(node)[2:]), int(str(node)[2:])], row])
                row += 1
                return {"PConst": str(node)}

        elif isinstance(node, sympy.Mul):
            args = node.args
            sub_structure = []

            for arg in args:
                sub_structure.append(traverse(arg))
            if len(sub_structure) > 2:
                return nest_structure(sub_structure, "mul")
            return {"mul": sub_structure}

        elif isinstance(node, sympy.Add):
            args = node.args
            sub_structure = []

            for arg in args:
                sub_structure.append(traverse(arg))
            if len(sub_structure) > 2:
                return nest_structure(sub_structure, "add")

            return {"add": sub_structure}

    structure_dict = traverse(expression)

    return structure_dict, ints, floats

# Example usage:
eq = "1.0*X_0 + X_1 + 1 + 2. + 3."
sp_eq = sp.parse_expr(eq)
structure = srepr(sp_eq)

structure_dict, ints, floats = parse_expression_structure(eq, simplify=True)

print(structure_dict)
import pdb;pdb.set_trace()
