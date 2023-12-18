import copy
from abc import abstractmethod

class Organism:
    def __init__(self):
        """
        Parameters
        ----------
        self : [Argument]
        genotype : [Argument]
        expression : [Argument]
        age :default: 0 [Argument]
        fitness :default: None [Argument]

        """
        self._genotype = None
        self._expression = None
        self._age = None
        self._fitness = None

    @property
    def genotype(self):
        """
        Parameters
        ----------
        self : [Argument]

        """
        return self._genotype

    @property
    def expression(self):
        """
        Parameters
        ----------
        self : [Argument]

        """
        return self._expression

    @property
    def age(self):
        """
        Parameters
        ----------
        self : [Argument]

        """
        return self._age

    @property
    def fitness(self):
        """
        Parameters
        ----------
        self : [Argument]

        """
        return self._fitness

    @genotype.setter
    def genotype(self, genotype):
        """
        Parameters
        ----------
        self : [Argument]
        genotype : [Argument]

        """
        self._genotype = genotype

    @expression.setter
    def expression(self, expression):
        """
        Parameters
        ----------
        self : [Argument]
        expression : [Argument]

        """
        self.expression = expression

    @age.setter
    def age(self, age):
        """
        Parameters
        ----------
        self : [Argument]
        age : [Argument]

        """
        self._age = age

    @fitness.setter
    def fitness(self, fitness):
        """
        Parameters
        ----------
        self : [Argument]
        fitness : [Argument]

        """
        self._fitness = fitness

    def copy(self):
        """
        Parameters
        ----------
        self : [Argument]

        """
        return copy.deepcopy(self)
    
    @abstractmethod
    def __str__(self):
        """
        Parameters
        ----------
        self : [Argument]

        """
        raise NotImplementedError
