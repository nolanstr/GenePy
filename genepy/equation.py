import warnings
import numpy as np
import torch

from .organism import Organism
from .genotype_from_phenotype import genotype_from_phenotype as gfp
from .phenotype_from_genotype import phenotype_from_genotype as pfg


class Equation(Organism):

    def __init__(self, genotype=None, phenotype=None):
        super().__init__()
        self._set_genetic_information(genotype, phenotype)

    def _set_genetic_information(self, genotype, phenotype):
        
        if (genotype is not None) and (phenotype is None):
            self._init_from_genotype(genotype)
        elif (genotype is None) and (phenotype is not None):
            self._init_from_phenotype(phenotype)
        else:
            warnings.warn(
        "Specrified both genotype and phenotype, initializing from phenotype")
            self._init_from_phenotype(phenotype)

    def _init_from_genotype(self, genotype):
        self._genotype = genotype
        self.set_number_of_constants()
        self.constants = torch.ones((1, self.number_of_constants))
        self._phenotype = pfg(genotype)

    def _init_from_phenotype(self, phenotype):
        structure_dict, ints, floats, genotype = gfp(phenotype)
        self._genotype = genotype
        self._phenotype = pfg(genotype)
    
    def set_number_of_constants(self):
        self.number_of_constants = np.sum(self.genotype[:,0]==1)
        self.number_of_iconstants = np.sum(self.genotype[:,0]==-1)


