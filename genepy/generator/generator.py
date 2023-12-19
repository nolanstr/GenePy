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
        """
        Parameters
        ----------
        self : object [Argument]
        genotype_size :default: 32 [Argument]
        X_dim :default: 1 [Argument]

        """
        self._genotype_size = genotype_size
        self._X_dim = X_dim
        self._operators = [0, 1]
        self._leaf_nodes = [0, 1]
        self._univariate_nodes = []
        self._bivariate_nodes = []

    def __call__(self, samples=1):
        """
        Parameters
        ----------
        self : object [Argument]
        samples :default: 1 [Argument]

        """
        if samples == 1:
            return self._generate_equation()
        return [self._generate_equation() for _ in range(samples)]

    def _generate_equation(self):
        """
        Parameters
        ----------
        self : object [Argument]

        """
        genotype = self.generate_genotype()
        equation = Equation(genotype=genotype)
        return equation
    
    def add_operators(self, operators):
        """
        Parameters
        ----------
        self : object [Argument]
        operators : [Argument]

        """
        for operator in operators:
            self.add_operator(operator)

    def add_operator(self, operator):
        """
        Parameters
        ----------
        self : object [Argument]
        operator : [Argument]

        """
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
        """
        Parameters
        ----------
        self : object [Argument]

        """
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
                self._operators, constants_count, current_stack_size=i
            )

        return genotype

    def _sample_gene(self, operators, constants_count, current_stack_size):
        """
        Parameters
        ----------
        self : object [Argument]
        operators : [Argument]
        constants_count : [Argument]
        current_stack_size : [Argument]

        """
        if len(operators) == len(self._operators) and \
                                    len(self._operators)>2:
            non_leaf_count = len(operators) - 2
            operator = np.random.choice(operators, 
                    p=[0.1, 0.1]+[0.8/non_leaf_count]*non_leaf_count)
        else:
            operator = np.random.choice(operators)

        if operator == 0:
            gene = [operator, constants_count, constants_count]
            constants_count += 1
        elif operator == 1:
            gene = [operator] + self._sample_X_idx()
        elif operator in self._univariate_nodes:
            node = self._sample_nodes(current_stack_size, n=1)
            gene = [operator] + node*2
        else:
            nodes = self._sample_nodes(current_stack_size, n=2)
            gene = [operator] + nodes

        return gene, constants_count

    def _sample_nodes(self, current_stack_size, n):
        """
        Parameters
        ----------
        self : object [Argument]
        current_stack_size : [Argument]
            n : [Argument]
        n : [Argument]
            n : [Argument]

        """
        #p = np.exp(current_stack_size)
        idxs = np.arange(0, current_stack_size)
        nodes = np.random.randint(0, current_stack_size, size=n).tolist()
        return nodes

    def _sample_X_idx(self):
        """
        Parameters
        ----------
        self : object [Argument]

        """
        idx = np.random.randint(0, self._X_dim)
        return [idx, idx]
