import sys
import numpy as np
import math
from scipy.stats import entropy
import pandas as pd

from functions import weighted_average


class CoEstimation():
    def __init__(self, dataset, alpha=0.1, M=100):
        self.dataset = dataset
        self.alpha = alpha
        self.M = M
        self.seed = None

    def get_token(self, token_of_surface):
        if isinstance(token_of_surface, Token):
            return token_of_surface
        if isinstance(token_of_surface, str):
            return self.dataset.get_token(token_of_surface)
        return None


    def initialize(self, seed):
        self.dataset.clear_score()
        self.set_seed(seed)


    def set_seed(self, token):
        self.seed = token
        token.set_current_score(1)


    def step1(self):
        for catalog in self.dataset.catalogs:
            for column in catalog.columns:
                if len(column.tokens) == 0:
                    column.s = 0
                else:
                    column.s = np.average([token.get_current_score()
                                           for token in column.tokens])


    def step2(self):
        for catalog in self.dataset.catalogs:
            distribution = np.array([column.s for column in catalog.columns])
            distribution_sum = np.sum(distribution)
            if distribution_sum == 0:
                catalog.M = 0
            else:
                catalog.M = np.exp(-entropy(distribution/distribution_sum, base=2))


    def step3(self):
        for token in self.dataset.surface2token.values():
            scores = np.array([column.s for column in token.occurrences])
            weights = np.array([np.exp(column.catalog.W)
                                for column in token.occurrences])
            E_ = weighted_average(scores, weights)
            new_score = (1 - self.alpha) * token.get_current_score() + self.alpha * E_
            token.add_new_score(new_score)


    def iterate(self):
        for _ in range(self.M):
            self.step1()
            self.step2()
            self.step3()


    def estimate_scores(self, seed):
        self.initialize(seed)
        self.iterate()


    def guess_target_tokens(self, item_description):
        max_maxE = max([token.maxE for token in item_description.tokens])
        return [token.surface for token in item_description.tokens
                if token.maxE == max_maxE]
        

class Dataset():
    def __init__(self):
        self.catalogs = []
        self.surface2token = {}


    def clear_score(self):
        for catalog in self.catalogs:
            catalog.clear_score()
        for token in self.surface2token.values():
            token.clear_score()


    def get_token(self, surface):
        if surface not in self.surface2token:
            self.surface2token[surface] = Token(surface)
        return self.surface2token[surface]


    def read_from_dict(self, dict_):
        for catalog_name in dict_:
            catalog = Catalog(self)
            self.catalogs.append(catalog)
            for surface_list in dict_[catalog_name]:
                catalog.add_item_description(surface_list)
        return self


class Catalog():
    def __init__(self, dataset):
        self.dataset = dataset
        self.item_descriptions = []
        self.columns = []
        self.W = 0


    def clear_score(self):
        self.W = 0


    def get_column(self, index_):
        for i in range(len(self.columns), index_ + 1):
            column = Column(self, i)
            self.columns.append(column)
        return self.columns[index_]

    
    def add_item_description(self, surface_list):
        description = ItemDescription(self, surface_list)
        self.item_descriptions.append(description)
        for i, token in enumerate(description.tokens):
            column = self.get_column(i)
            column.contains(token)
            token.occurs_in(column)


class ItemDescription():
    def __init__(self, catalog, surface_list):
        self.catalog = catalog
        self.tokens = []
        for surface in surface_list:
            token = self.catalog.dataset.get_token(surface)
            self.tokens.append(token)


    def get_tokens(self, surface=True):
        if surface:
            return [token.surface for token in self.tokens]
        else:
            return self.tokens


    def guess_target_tokens(self):
        max_maxE = max([token.maxE for token in self.tokens])
        return [token.surface for token in self.tokens if token.maxE == max_maxE]


class Column():
    def __init__(self, catalog, index_):
        self.catalog = catalog
        self.index_ = index_
        self.tokens = []
        self.s = 0

    def __eq__(self, obj):
        if isinstance(obj, Column):
            return self.catalog == obj.catalog and self.index_ == obj.index_
        return False

    
    def __hash__(self):
        return self.index_


    def contains(self, token):
        self.tokens.append(token)



class Token():
    def __init__(self, surface):
        self.surface = surface
        #self.columns = set()
        self.occurrences = []
        self.maxE = 0
        self.E_history = [0]


    def __eq__(self, obj):
        if isinstance(obj, Token):
            return self.surface == obj.surface
        return False

    
    def __hash__(self):
        return hash(self.surface)


    def __str__(self):
        return self.surface

    
    def occurs_in(self, column):
        self.occurrences.append(column)


    def clear_score(self):
        self.maxE = 0
        self.E_history = [0]


    def update_E(self, score):
        if self.maxE < score:
            self.maxE = score


    def get_current_score(self):
        return self.E_history[-1]

    
    def set_current_score(self, score):
        self.E_history[-1] = score
        self.update_E(score)


    def add_new_score(self, score):
        self.E_history.append(score)
        self.update_E(score)


    def get_score_history(self):
        return self.E_history
