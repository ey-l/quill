import numpy as np
from typing import List

class TestPartition:
    
    def __init__(self, ID, method, bins, distribution, kl_d, l2_norm, gpt_semantics, gold_standard, attribute, binned_values=None):
        self.ID = ID
        self.bins = bins
        self.distribution = distribution
        self.KLDiv = kl_d
        self.l2_norm = l2_norm
        self.method = method
        self.gpt_semantics = gpt_semantics
        self.buckets = None
        self.method = None
        self.binned_values = binned_values
        self.attribute = attribute
        self.f_time = []
        self.utility = 0
        self.gold_standard = gold_standard
    
    def __repr__(self) -> str:
        return f'Partition({self.bins}, {self.method}, {self.KLDiv}, {self.l2_norm})'

    def __eq__(self, __value: object) -> bool:
        return self.bins == __value.bins
    
    def get_semantic_score(self, metric:str): 
        """
        Get the semantic score for the given metric.
        :param metric: The metric to use.
        """
        if metric == 'KLDiv': return self.KLDiv
        elif metric == 'l2_norm': return self.l2_norm
        elif metric == 'gpt_semantics': return self.gpt_semantics
        else: raise ValueError('Invalid metric.')
    
    def set_utility(self, utility:float):
        """
        Set the utility of the partition.
        :param utility: The utility value to set.
        """
        self.utility = utility
        #if self.utility == 0: raise ValueError('Utility not set.')
        if self.utility < 0: raise ValueError('Utility cannot be negative.')


class TestSearchSpace:
    """
    Class for search space.
    """
    def __init__(self, candidate_df=None, attribute=None, decimal_place=2, max_bin_size=10):
        self.candidate_df = candidate_df
        self.attribute = attribute
        self.decimal_place = decimal_place
        self.max_bin_size = max_bin_size
        self.candidates = self._create_candidates_from_df()
    
    def _create_candidates_from_df(self) -> List[TestPartition]:
        """
        Create candidates from the candidate dataframe.
        """
        candidates = []
        bins_list = []
        for i in range(self.candidate_df.shape[0]):
            row = self.candidate_df.iloc[i]
            bins = row['bins']
            bins = np.fromstring(bins[1:-1], dtype=float, sep=' ')
            # Round the bins to the decimal place of the attribute
            bins = np.unique(np.round(bins, decimals=self.decimal_place))
            # Filter by max_bin_size, continue if row['method'] is not 'gold-standard' nor 'equal-width'
            if len(bins)-1 > self.max_bin_size and row['method'] != 'gold-standard': continue

            #binned_values = row['binned_values']
            #binned_values = np.fromstring(binned_values[1:-1], dtype=float, sep=' ')
            values = row['distribution']
            values = np.fromstring(values[1:-1], dtype=float, sep=' ')
            method = row['method']
            if method == 'gold-standard':
                partition = TestPartition(ID=row['ID'], method=row['method'], bins=bins, distribution=values, kl_d=float(row['kl_d']), l2_norm=float(row['l2_norm']), gpt_semantics=float(row['gpt_prompt']), gold_standard=True, attribute=self.attribute)
            else: partition = TestPartition(ID=row['ID'], method=row['method'], bins=bins, distribution=values, kl_d=float(row['kl_d']), l2_norm=float(row['l2_norm']), gpt_semantics=float(row['gpt_prompt']), gold_standard=False, attribute=self.attribute)
            
            # Check if bins already exist in candidates
            for b in bins_list:
                if np.array_equal(bins, b): break
            else: 
                # Only add the partition if the bins do not exist
                candidates.append(partition)
                bins_list.append(bins)
        return candidates
    
    def get_candidate(self, ID:int) -> TestPartition:
        """
        Get the candidate with the given ID.
        """
        for candidate in self.candidates:
            if candidate.ID == ID: return candidate
        return None