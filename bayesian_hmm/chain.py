#!/usr/bin/env python3
"""
The Chain object stores a single emission series.
It has methods to initialise the object, resample the latent states, and some convenient
printing methods.
"""
# Support typehinting.
from __future__ import annotations
from typing import (Collection,
    Sequence,
    Dict,
    Any,
    Union,
    List,
    Tuple,
    Set,
    Optional,
    Sized,
    Iterable)

import numpy as np
import random
import copy

# shorthand for numeric types
Numeric = Union[int, float]

# oft-used dictionary initializations with shorthands
DictStrNum = Dict[Optional[str], Numeric]
InitDict = DictStrNum
DictStrDictStrNum = Dict[Optional[str], DictStrNum]
NestedInitDict = DictStrDictStrNum


# chain stores a single markov emission sequence plus associated latent variables
class Chain(object):
    
    """
    Create a Hidden Markov Chain of latent sequences for an observed emission sequence.
    :param sequence: iterable containing observed emissions.
    """

    def __init__(self, sequence: List[Optional[str]]) -> None:
        
        """
        initialise & store emission sequences and list of Nones as latent sequences
        """
        
        # self.emission_sequence as deep copy of original sequence object
        self.emission_sequence: List[Optional[str]] = copy.deepcopy(sequence)
        # self.latent_sequence begins as list of None objects for each item in sequence
        self.latent_sequence: List[Optional[str]] = [None for _ in sequence]
        # self.T as length of emission_sequence
        self.T = len(self.emission_sequence)
        # keep flag to track initialisation
        self._initialised_flag = False

    def __len__(self) -> int:
        
        return self.T

    @property
    def initialised_flag(self) -> bool:
        
        """
        Test whether a chain is initialised.
        :return: bool
        """
        
        return self._initialised_flag

    @initialised_flag.setter
    def initialised_flag(self, value: bool) -> None:
        
        if value is True:
            raise RuntimeError("Chain must be initialised through initialise_chain method")
        elif value is False:
            self._initialised_flag = False
        else:
            raise ValueError("initialised flag must be Boolean")

    def __repr__(self) -> str:
        
        return "<bayesian_hmm.Chain, size {0}>".format(self.T)

    def __str__(self, print_len: int = 15) -> str:
        
        print_len = min(print_len - 1, self.T - 1)
        return "bayesian_hmm.Chain, size={T}, seq={s}".format(T=self.T,
                                                              s=["{s}:{e}".format(s=s, e=e)
                                                                 for s, e in zip(self.latent_sequence, self.emission_sequence)][:print_len] + ["..."],)

    def tabulate(self) -> np.array:
        
        """
        Convert the latent and emission sequences into a single numpy array.
        :return: numpy array with shape (n, 2), where n is the length of the chain
        e.g.: [[state_name,emission],[state_name,emission]]
        """
        
        return np.column_stack((copy.copy(self.latent_sequence), copy.copy(self.emission_sequence)))

    def initialise(self, states: Sequence) -> None:
        
        """
        Randomly initialise latent variables in chain.
        Typically called directly from an HDPHMM object.
        :param states: set of states to sample from
        :return: None
        """
        
        # overwrite list of Nones with random choices from states argument
        self.latent_sequence = random.choices(states, k=self.T)

        # update _initialised_flag to True
        self._initialised_flag = True

    def neglogp_chain(self,
                      p_initial: InitDict,
                      p_emission: NestedInitDict,
                      p_transition: NestedInitDict,) -> Numeric:
        
        """
        Negative log likelihood of the chain, using the given parameters.
        Usually called with parameters given by the parent HDPHMM object.
        :param p_initial: dict, initial probabilities
        :param p_emission: dict, emission probabilities
        :param p_transition: dict, transition probabilities
        :return: float
        """
        
        # edge case: zero-length sequence
        if self.T == 0:
            return 0

        # np.prod([])==1, so this is safe
        
        # add log prob of initial state and log prob of emission given state for first emission
        p_start = np.log(p_initial[self.latent_sequence[0]]) + np.log(p_emission[self.latent_sequence[0]][self.emission_sequence[0]])

        # for each subsequent timestep, add emission probability given state and transition probability from prev timestep to current
        p_remainder = [np.log(p_emission[self.latent_sequence[t]][self.emission_sequence[t]])
            + np.log(p_transition[self.latent_sequence[t - 1]][self.latent_sequence[t]])
            for t in range(1, self.T)]

        # sum up the two objects and negate
        return -(p_start + sum(p_remainder))

    @staticmethod
    def resample_latent_sequence(sequences: Tuple[List[str], List[str]],
                                states: Set[str],
                                p_initial: InitDict,
                                p_emission: NestedInitDict,
                                p_transition: NestedInitDict) -> List[str]:
        
        """
        Resample the latent sequence of a chain. This is usually called by another
        method or class, rather than directly. It is included to allow for
        multithreading in the resampling step.
        :param sequences: tuple(list, list), an emission sequence and latent beam
        variables
        :param states: set, states to choose from
        :param p_initial: dict, initial probabilities
        :param p_emission: dict, emission probabilities
        :param p_transition: dict, transition probabilities
        :return: list, resampled latent sequence
        """

        # extract size information
        emission_sequence, latent_sequence = sequences
        seqlen = len(emission_sequence)

        # edge case: zero-length sequence
        if seqlen == 0:
            return []

        # list of initial and transition probabilities using current latent sequence
        temp_p_transition = [p_initial[latent_sequence[0]]] + [p_transition[latent_sequence[t]][latent_sequence[t + 1]] 
        for t in range(seqlen - 1)]

        # beam sampling u's for each pi__(s_t-1,s_t) using probs of current latent sequence
        auxiliary_vars = [np.random.uniform(0, p) for p in temp_p_transition]

        # initialise historical P(s_t | u_{1:t}, y_{1:t}) and latent sequence
        p_history: List[Dict[str, Numeric]]
        p_history = [dict()] * seqlen
        latent_sequence = [str()] * seqlen

        ### RETURN TO THIS ###: Note that Fox (2009) starts with a backward recursion
        ### FORWARD RECURSION ###
        # compute probability emissions at t=0 overall possible states if the init state probs
        # are greater than the aux_var u probabilities
        p_history[0] = {s: p_initial[s] * p_emission[s][emission_sequence[0]] if p_initial[s] > auxiliary_vars[0] else 0 for s in states}

        # for remaining states, probabilities are function of emission and transition
        for t in range(1, seqlen):
            
            # p_temp for s2 = sum(p(s1)) for all states s1 * p(o_t2|s2) for all states s2 where p(s2|s1) > u_t2
            # Note: We're using the actual emissions from the chain for t2
            # Note: No need for an 'else 0' since we just don't include instances where transition is less than aux to the sum
            ### RETURN TO THIS ### need to multiply previous state prob by transition prob
            p_temp = {s2: sum(p_history[t - 1][s1] * p_transition[s1][s2] for s1 in states if p_transition[s1][s2] > auxiliary_vars[t])
            * p_emission[s2][emission_sequence[t]] for s2 in states}
            
            # we then marginlalize the resulting p_temp and p_temp becomes p_history for that timestep
            p_temp_total = sum(p_temp.values())
            p_history[t] = {s: p_temp[s] / p_temp_total for s in states}

        # choose ending state based on probability weights
        latent_sequence[seqlen - 1] = random.choices(tuple(p_history[seqlen - 1].keys()),
                                        weights=tuple(p_history[seqlen - 1].values()), k=1)[0]

        ### BACKWARD RECURSION ###
        # work backwards to compute new latent sequence starting at the penultimate timestamp
        for t in range(seqlen - 2, -1, -1):
            
            # p_temp for s1 = p(s1) * p(s2_actual|s1) if p(s2_actual|s1) > u_t2 for all states 1
            # Note: We're using the actual latent states (after resampling) from the chain for s2 to update p(s1)
            p_temp = {s1: p_history[t][s1] * p_transition[s1][latent_sequence[t + 1]]
                if p_transition[s1][latent_sequence[t + 1]] > auxiliary_vars[t + 1] else 0 for s1 in states}
            
            # choose new latent state for current timestep based on p(s1) for all s1
            latent_sequence[t] = random.choices(tuple(p_temp.keys()), weights=tuple(p_temp.values()), k=1)[0]

        # latent sequence now completely filled
        return latent_sequence
