import torch
import numpy as np
import re
import sympy
import sympy.parsing as sp
from sympy import symbols, srepr, sympify, fraction
from sympy.core.traversal import postorder_traversal, preorder_traversal
from sympy.core.numbers import Float, Integer, Rational
from sympy import exp, log, sin, cos, tan, asin, acos, atan, sinh, cosh, tanh

op_dict = {
    "pconst": -1,
    "cconst": 0,
    "var": 1,
    "add": 2,
    "sub": 3,
    "mul": 4,
    "div": 5,
    "pow": 6,
    "sqrt": 7,
    "exp": 8,
    "log": 9,
    "sin": 10,
    "cos": 11,
    "tan": 12,
    "asin": 13,
    "acos": 14,
    "atan": 15,
    "sinh": 16,
    "cosh": 17,
    "tanh": 18,
}
"""
Curently, sympy does not support a sympy.Sub expression and thus all expressions
with subtraction are replace with addition and multiplication. I.e. "1-2X" gives
sympy.Add(1, symp.Mul(2,X)).
This is consistent with divide (replaced by multiply) and sqrt (replaced by
power).
"""


def genotype_from_phenotype(structure, simplify=True):
    """
    Parameters
    ----------
    structure : [Argument]
    simplify :default: True [Argument]

    """
    expression = make_sympy_expression(structure)
    print(expression)
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
    print(f"kjh: {expression}")
    genes = []
    row = 0

    def traverse(node):
        """
        Parameters
        ----------
        node : [Argument]

        """

        nonlocal genes, row

        if isinstance(node, sympy.Symbol):
            
            if "X" in str(node):
                gene = update_genes([1, int(str(node)[2:]), int(str(node)[2:])])
                return {"var": {"symbol": str(node), "gene": gene}}
            elif "C" in str(node):
                gene = update_genes([0, int(str(node)[2:]), int(str(node)[2:])])
                return {"const": {"symbol": str(node), "gene": gene}}
            elif "R" in str(node) or "I" in str(node):
                if str(node).count("_") == 2:
                    n, d = str(node).split("_")[1:]
                    value = eval(n) / eval(d)
                else:
                    value = eval(str(node)[2:])
                gene = update_genes([-1, value, value])
                return {"pconst": {"symbol": str(node), "gene": gene}}
            else:
                gene = update_genes([op_dict[str(node)], row-1, row-1])
                return {get_op(node): {"symbol": str(node), "gene": gene}}

        else:
            args = node.args
            sub_structure = []
            for arg in args:
                sub_structure.append(traverse(arg))
            op = get_op(node)
            return nest_structure(sub_structure, op)

    def nest_structure(structure, op):
        """
        Parameters
        ----------
        structure : [Argument]
        op : [Argument]

        """
        nonlocal genes, row
        if len(structure) == 1:

            if unary_op(op):
                row_u = structure[0][next(iter(structure[0]))]["gene"][1]
                gene = update_genes([op_dict[op], row_u, row_u])
                return {
                    op: {
                        "symbol": structure,
                        "gene": gene,
                    }
                }
            else:
                return structure[0]

        elif len(structure) == 2:
            row1 = structure[0][next(iter(structure[0]))]["gene"][1]
            row2 = structure[1][next(iter(structure[1]))]["gene"][1]
            gene = update_genes([op_dict[op], row1, row2])
            return {
                op: {
                    "symbol": structure,
                    "gene": gene,
                }
            }

        else:
            rows = [s[next(iter(s))]["gene"][1] for s in structure]
            for i in range(len(rows)-1):
                if i == 0:
                    row1 = rows[i]
                    row2 = rows[i+1]
                    gene = update_genes([op_dict[op], row1, row2])
                else:
                    row1 = row-1
                    row2 = rows[i+1]
                    gene = update_genes([op_dict[op], row1, row2])
            return {
                op: {
                    "symbol": structure,
                    "gene": gene,
                }
            }

    def update_genes(new_gene):
        nonlocal genes, row
        for i, gene in enumerate(genes):
            if new_gene == gene[0]:
                return gene
        genes.append([new_gene, row])
        row += 1
        return genes[-1]

    structure_dict = traverse(expression)
    genes_order = [gene[1] for gene in genes]
    genes = np.vstack([np.array(gene[0], dtype=object) for gene in genes])[
        np.argsort(genes_order), :
    ]
    
    return structure_dict, ints, floats, genes


def unary_op(op):
    return op in ["e", "exp", "log", "sin", "cos", "tan", 
                    "asin", "acos", "atan", "sinh", "cosh", "tanh"]

def make_sympy_expression(string):
    """
    Parameters
    ----------
    string : [Argument]

    """
    if isinstance(string, sympy.Expr):
        return string
    pattern = r"[A-Za-z_]\w*"
    pre_variables = list(set(re.findall(pattern, string)))
    variables = []
    for var in pre_variables:
        if not unary_op(var):
            variables.append(var)
    symbols = [sympy.Symbol(v) for v in variables]
    sdict = {str(v):sympy.Symbol(v) for v in variables}

    return sympify(string, evaluate=False, locals=sdict)


def extract_constants(expression):
    """
    Parameters
    ----------
    expression : [Argument]

    """
    ints = {}
    floats = {}
    for term in expression.atoms():
        if term.is_constant():
            if isinstance(term, Integer):
                ints[f"I_{term}"] = int(term)
            elif isinstance(term, Float):
                floats[f"C_{len(floats)}"] = float(term)
            elif isinstance(term, Rational):
                n, d = fraction(term)
                ints[f"R_{n}_{d}"] = term
            else:
                raise NotImplementedError("extract constant unknown")
    return ints, floats


def replace_constants(expression, consts_dict):
    replacement_symbols = [sympy.Symbol(symbol) for symbol in consts_dict.keys()]
    replaced_expression = expression.subs(
        list(zip(consts_dict.values(), replacement_symbols))
    )
    """
    Parameters
    ----------
    expression : [Argument]
    consts_dict : [Argument]

    """
    return replaced_expression


def replace_symbols(expression, consts_dict):
    """
    Parameters
    ----------
    expression : [Argument]
    consts_dict : [Argument]

    """
    replacement_symbols = [sympy.Symbol(symbol) for symbol in consts_dict.keys()]
    replacement_values = [
        sympy.UnevaluatedExpr(symbol) for symbol in consts_dict.values()
    ]
    replaced_expression = expression.subs(
        list(zip(replacement_symbols, replacement_values)), evaluate=False
    )
    return replaced_expression


def get_op(node):
    """
    Parameters
    ----------
    node : [Argument]

    """
    if isinstance(node, sympy.Add):
        return "add"
    # elif isinstance(node, sympy.Sub):
    #    return "sub"
    elif isinstance(node, sympy.Mul):
        return "mul"
    # elif isinstance(node, sympy.Div):
    #    return "div"
    elif isinstance(node, sympy.Pow):
        return "pow"
    # elif isinstance(node, sympy.sqrt):
    #    return "sqrt"
    elif str(node) == "exp" or isinstance(node, sympy.exp):
        return "exp"
    elif str(node) == "log" or isinstance(node, sympy.log):
        return "log"
    elif str(node) == "sin" or isinstance(node, sympy.sin):
        return "sin"
    elif str(node) == "cos" or isinstance(node, sympy.cos):
        return "cos"
    elif str(node) == "tan" or isinstance(node, sympy.tan):
        return "tan"
    elif str(node) == "asin" or isinstance(node, sympy.asin):
        return "asin"
    elif str(node) == "acos" or isinstance(node, sympy.acos):
        return "acos"
    elif str(node) == "atan" or isinstance(node, sympy.atan):
        return "atan"
    elif str(node) == "sinh" or isinstance(node, sympy.sinh):
        return "sinh"
    elif str(node) == "cosh" or isinstance(node, sympy.cosh):
        return "cosh"
    elif str(node) == "tanh" or isinstance(node, sympy.tanh):
        return "tanh"
    else:
        raise NotImplementedError("operator not implemented/bad phenotype description")
