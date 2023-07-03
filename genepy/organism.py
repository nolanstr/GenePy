import copy
from abc import abstractmethod

class Organism:
    def __init__(self):
        """
        Parameters
        ----------
        self : [Argument]
        genotype : [Argument]
        phenotype : [Argument]
        age :default: 0 [Argument]
        fitness :default: None [Argument]

        """
        self._genotype = None
        self._phenotype = None
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
    def phenotype(self):
        """
        Parameters
        ----------
        self : [Argument]

        """
        return self._phenotype

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

    @phenotype.setter
    def phenotype(self, phenotype):
        """
        Parameters
        ----------
        self : [Argument]
        phenotype : [Argument]

        """
        self.phenotype = phenotype

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
