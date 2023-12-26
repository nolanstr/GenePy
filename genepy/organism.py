import copy
from abc import abstractmethod


class Organism:
    def __init__(self):
        """
        Parameters
        ----------
        self : object [Argument]
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
        self : object [Argument]

        """
        return self._genotype

    @property
    def expression(self):
        """
        Parameters
        ----------
        self : object [Argument]

        """
        return self._expression

    @property
    def age(self):
        """
        Parameters
        ----------
        self : object [Argument]

        """
        return self._age

    @property
    def fitness(self):
        """
        Parameters
        ----------
        self : object [Argument]

        """
        return self._fitness

    @genotype.setter
    def genotype(self, genotype):
        """
        Parameters
        ----------
        self : object [Argument]
        genotype : [Argument]

        """
        self._genotype = genotype

    @expression.setter
    def expression(self, expression):
        """
        Parameters
        ----------
        self : object [Argument]
        expression : [Argument]

        """
        self._expression = expression

    @age.setter
    def age(self, age):
        """
        Parameters
        ----------
        self : object [Argument]
        age : [Argument]

        """
        self._age = age

    @fitness.setter
    def fitness(self, fitness):
        """
        Parameters
        ----------
        self : object [Argument]
        fitness : [Argument]

        """
        self._fitness = fitness

    @abstractmethod
    def copy(self):
        """
        Parameters
        ----------
        self : object [Argument]

        """
        raise NotImplementedError

    @abstractmethod
    def __str__(self):
        """
        Parameters
        ----------
        self : object [Argument]

        """
        raise NotImplementedError
