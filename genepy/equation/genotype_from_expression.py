import torch
import numpy as np
import re

C_pattern = re.compile(r"C_\d+")
I_pattern = re.compile(r"I_\d+")
import sympy
import sympy.parsing as sp
from sympy import symbols, srepr, sympify, fraction
from sympy.core.traversal import postorder_traversal, preorder_traversal
from sympy.core.numbers import Float, Integer, Rational
from sympy import sqrt, exp, log, sin, cos, tan, asin, acos, atan, sinh, cosh, tanh

sympy_operators = [sqrt, exp, log, sin, cos, tan, asin, acos, atan, sinh, cosh, tanh]

op_dict = {
    "pconst": -1,
    "cconst": 0,
    "var": 1,
    "add": 2,
    "sub": 3,
    "mul": 4,
    "div": 5,
    "pow": 6,
    "square": 7,
    "sqrt": 8,
    "exp": 9,
    "log": 10,
    "sin": 11,
    "cos": 12,
    "tan": 13,
    "asin": 14,
    "acos": 15,
    "atan": 16,
    "sinh": 17,
    "cosh": 18,
    "tanh": 19,
}
"""
Curently, sympy does not support a sympy.Sub expression and thus all expressions
with subtraction are replace with addition and multiplication. I.e. "1-2X" gives
sympy.Add(1, symp.Mul(2,X)).
This is consistent with divide (replaced by multiply) and sqrt (replaced by
power).
"""


def update_dict(DICT, _DICT):
    """
    Parameters
    ----------
    DICT : [Argument]
    _DICT : [Argument]

    """
    update = False
    for key in _DICT.keys():
        if key in DICT.keys():
            pass
        else:
            DICT[key] = _DICT[key]
            update = True
    return DICT, update


def update_ints_floats(ints, floats, _ints, _floats):
    """
    Parameters
    ----------
    ints : [Argument]
    floats : [Argument]
    _ints : [Argument]
    _floats : [Argument]

    """
    ints, ints_update = update_dict(ints, _ints)
    floats, floats_update = update_dict(floats, _floats)
    return ints, floats, ints_update or floats_update


def check_unknowns(expression):
    """
    Parameters
    ----------
    expression : [Argument]

    """

    C_values = list(set(C_pattern.findall(expression)))
    I_values = list(set(I_pattern.findall(expression)))

    ints = {I_value: 1 for I_value in I_values}
    floats = {C_value: 1.0 for C_value in C_values}

    return ints, floats


def genotype_from_expression(structure, simplify=True):
    """
    Parameters
    ----------
    structure : [Argument]
    simplify :default: True [Argument]

    """

    if simplify:
        expression, sqrt_check = make_sympy_expression(structure)
        if "zoo" in str(expression) or "nan" in str(expression):
            return None, {"I_0": np.nan}, {}, [[-1, 0, 0]]
        ints, floats = check_unknowns(str(expression))
        if sqrt_check:
            ints[str(sqrt_check)] = 0.5
        check = True
        it = 0
        while check:
            expression = sympify(expression, evaluate=False)
            _ints, _floats = extract_constants(expression, len(ints), len(floats))
            ints, floats, check = update_ints_floats(ints, floats, _ints, _floats)
            expression = replace_constants(expression, ints)
            expression = replace_constants(expression, floats)
            it += 1
            if it > 20:
                print("Failed to generate genotype from expression!")
                return None, {"I_0": np.nan}, {}, [[-1, 0, 0]]

        if isinstance(sympy.simplify(expression), int):
            return None, {"I_0": sympy.simplfiy(expression)}, {}, [[-1, 0, 0]]
        if isinstance(sympy.simplify(expression), float):
            return None, {}, {"C_0": sympy.simplfiy(expression)}, [[0, 0, 0]]

    else:
        ints, floats = extract_constants(expression, 0, 0)
        expression = replace_constants(expression, ints)
        expression = replace_constants(expression, floats)

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
                gene = update_genes([-1, int(str(node)[2:]), int(str(node)[2:])])
                return {"pconst": {"symbol": str(node), "gene": gene}}
            else:
                gene = update_genes([op_dict[str(node)], row - 1, row - 1])
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
            for i in range(len(rows) - 1):
                if i == 0:
                    row1 = rows[i]
                    row2 = rows[i + 1]
                    gene = update_genes([op_dict[op], row1, row2])
                else:
                    row1 = row - 1
                    row2 = rows[i + 1]
                    gene = update_genes([op_dict[op], row1, row2])
            return {
                op: {
                    "symbol": structure,
                    "gene": gene,
                }
            }

    def update_genes(new_gene):
        """
        Parameters
        ----------
        new_gene : [Argument]

        """
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

    genes = clean_constants_in_genes(genes)
    return structure_dict, ints, floats, genes


def clean_constants_in_genes(genes):
    """
    Parameters
    ----------
    genes : [Argument]

    """
    constant_idxs = np.argwhere(genes[:, 0].flatten() == 0).flatten()
    constant_tags = np.unique(genes[constant_idxs, 1])
    for i, constant_tag in enumerate(constant_tags):
        constant_tag_idxs = np.argwhere(genes[constant_idxs, 1] == constant_tag)
        genes[constant_idxs[constant_tag_idxs], 1:] = i

    Iconstant_idxs = np.argwhere(genes[:, -1].flatten() == 0).flatten()
    Iconstant_tags = np.unique(genes[Iconstant_idxs, 1])
    for i, Iconstant_tag in enumerate(Iconstant_tags):
        Iconstant_tag_idxs = np.argwhere(genes[Iconstant_idxs, 1] == Iconstant_tag)
        genes[Iconstant_idxs[Iconstant_tag_idxs], 1:] = i

    return genes


def unary_op(op):
    """
    Parameters
    ----------
    op : [Argument]

    """
    return op in [
        "square",
        "sqrt",
        "e",
        "exp",
        "log",
        "sin",
        "cos",
        "tan",
        "asin",
        "acos",
        "atan",
        "sinh",
        "cosh",
        "tanh",
    ]


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
    sdict = {str(v): sympy.Symbol(v) for v in variables}
    expr = sympify(string, evaluate=False, locals=sdict)
    if "sqrt" in pre_variables:
        I_terms = sum(["R_" in key for key in sdict.keys()])
        sqrt_symbol = sympy.Symbol(f"R_{I_terms}")
        expr = replace_sqrt(expr, sqrt_symbol)
        return expr, sqrt_symbol
    return expr, False


def replace_sqrt(expr, value):
    """
    Parameters
    ----------
    expr : [Argument]
    value : [Argument]

    """
    if expr.is_Pow:
        base = replace_sqrt(expr.base, value)
        if expr.exp == (1 / 2):
            exp = value
        else:
            exp = replace_sqrt(expr.exp, value)
        return base**exp
    elif expr.is_Function and expr.func == sqrt:
        return expr.args[0] ** value
    elif hasattr(expr, "args"):
        if len(expr.args) == 0:
            return expr
        return expr.func(*[replace_sqrt(arg, value) for arg in expr.args])
    else:
        return expr


def extract_constants(expression, ints_offset, floats_offset):
    """
    Parameters
    ----------
    expression : [Argument]
    ints_offset : [Argument]
    floats_offset : [Argument]

    """
    ints = {}
    floats = {}

    for term in expression.atoms():
        if term.is_constant():
            if isinstance(term, Integer):
                ints[f"I_{len(ints)+ints_offset}"] = int(term)
            elif isinstance(term, Float):
                floats[f"C_{len(floats)+floats_offset}"] = float(term)
            elif isinstance(term, Rational):
                n, d = fraction(term)
                ints[f"R_{len(ints)+ints_offset}"] = float(term)
            else:
                raise NotImplementedError("extract constant unknown")

    return ints, floats


def replace_constants(expression, consts_dict):
    """
    Parameters
    ----------
    expression : [Argument]
    consts_dict : [Argument]

    """
    replacement_symbols = [sympy.Symbol(symbol) for symbol in consts_dict.keys()]
    replaced_expression = expression.subs(
        dict(zip(consts_dict.values(), replacement_symbols)), evaluate=False
    )
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
    elif isinstance(node, sympy.Mul):
        return "mul"
    elif isinstance(node, sympy.Pow):
        return "pow"
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
        raise NotImplementedError("operator not implemented/bad expression description")
