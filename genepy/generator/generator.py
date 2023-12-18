import numpy as np

from ..equation import Equation

UNIVARIATE_DICT = {
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

BIVARIATE_DICT = {
    "add": 2,
    "sub": 3,
    "mult": 4,
    "div": 5,
    "pow": 6,
}

UNIVARIATE_OPS = list(UNIVARIATE_DICT.keys())
BIVARIATE_OPS = list(BIVARIATE_DICT.keys())
SUPPORTED_OPS = BIVARIATE_OPS + UNIVARIATE_OPS


class Generator:
    def __init__(self, genotype_size=32, X_dim=1):
        self._genotype_size = genotype_size
        self._X_dim = X_dim
        self._operators = [0, 1]
        self._leaf_nodes = [0, 1]
        self._univariate_nodes = []
        self._bivariate_nodes = []

    def __call__(self, samples=1):
        if samples == 1:
            return self._generate_equation()
        return [self._generate_equation() for _ in range(samples)]

    def _generate_equation(self):
        genotype = self.generate_genotype()
        equation = Equation(genotype=genotype)
        return equation

    def add_operator(self, operator):
        if operator in UNIVARIATE_OPS:
            self._operators.append(UNIVARIATE_DICT[operator.lower()])
            self._univariate_nodes.append(UNIVARIATE_DICT[operator.lower()])
        elif operator in BIVARIATE_OPS:
            self._operators.append(BIVARIATE_DICT[operator.lower()])
            self._bivariate_nodes.append(BIVARIATE_DICT[operator.lower()])
        else:
            raise ValueError("Operator not supported!")
        self._operators = list(set(self._operators))

    def generate_genotype(self):
        genotype = np.zeros((self._genotype_size, 3)).astype(int)
        constants_count = 0
        genotype[0, :], constants_count = self._sample_gene(
            self._leaf_nodes, constants_count, current_stack_size=0
        )
        genotype[1, :], constants_count = self._sample_gene(
            self._leaf_nodes + self._univariate_nodes,
            constants_count,
            current_stack_size=1,
        )

        for i in range(2, self._genotype_size):
            genotype[i, :], constants_count = self._sample_gene(
                self._operators, constants_count, current_stack_size=2
            )

        return genotype

    def _sample_gene(self, operators, constants_count, current_stack_size):
        operator = np.random.choice(operators)
        if operator == 0:
            gene = [operator, constants_count, constants_count]
            constants_count += 1
        elif operator == 1:
            gene = [operator] + self._sample_X_idx()
        else:
            nodes = np.random.randint(0, current_stack_size, size=2)
            gene = [operator] + nodes.tolist()
        return gene, constants_count

    def _sample_X_idx(self):
        idx = np.random.randint(0, self._X_dim)
        return [idx, idx]
