import sympy
import sympy.parsing as sp
from sympy import symbols, srepr, sympify

from sympy.core.numbers import Float, Integer

def extract_constants(expression):
    constants = []

    for term in expression.as_ordered_terms():
        if term.is_constant():
            constants.append(term)

    return constants

def replace_constants(expression, symbols):
    constants = extract_constants(expression)

    # Create symbols for replacement
    replacement_symbols = [sympy.Symbol(symbol) for symbol in symbols]

    # Replace constants with symbols in the expression
    replaced_expression = expression.subs(list(zip(constants, replacement_symbols)))

    return replaced_expression

def parse_expression_structure(structure, simplify=False):
    expression = sympify(structure, evaluate=False)
    constants = extract_constants(expression)
    import pdb;pdb.set_trace()
    expression = sympify(structure, evaluate=simplify)
    import pdb;pdb.set_trace()
    def traverse(node):
        if isinstance(node, sympy.Symbol):
            return {"var": str(node)}
        elif isinstance(node, int) or isinstance(node, float):
            import pdb;pdb.set_trace()
            #return {"const": str(node)}
            return {"const": repr(node)}
        elif isinstance(node, sympy.Mul):
            args = node.args
            sub_structure = []

            for arg in args:
                sub_structure.append(traverse(arg))

            return {"mul": sub_structure}

        elif isinstance(node, sympy.Add):
            args = node.args
            sub_structure = []

            for arg in args:
                sub_structure.append(traverse(arg))

            return {"add": sub_structure}
        import pdb;pdb.set_trace()

    structure_dict = traverse(expression)

    return structure_dict

# Example usage:
eq = "(X_0 + 2 + 1.0 * X_1) + 3."
sp_eq = sp.parse_expr(eq)
structure = srepr(sp_eq)

structure_dict = parse_expression_structure(eq)

print(structure_dict)
import pdb;pdb.set_trace()
