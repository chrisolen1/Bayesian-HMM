#!/usr/bin/env python3
"""
Helper functions for the Bayesian-HMM package. Should not be called directly by the
user.
"""
# Support typehinting.
from __future__ import annotations
from typing import Union, Generator, Iterator, Dict, Optional

import numpy as np
import random
import itertools
import string

# Shorthand for a numeric type.
Numeric = Union[int, float]


# used to give human-friendly labels to states as they are created
def label_generator(labels: str = string.ascii_lowercase) -> Generator[str, None, None]:
    """
    :param labels: set of labels to choose from. Should not be numeric.
    :return: a generator which  yields unique labels of the form
    a, b, ..., z, a1, b1, ...
    """
    x, y, z = 0, 0, ""
    while True:
        if x < len(labels):
            yield labels[x] + z
            x += 1
        if x == len(labels):
            y += 1
            z = str(y)
            x = 0


# used to choose from new states after resampling latent states
def dirichlet_process_generator(alpha: Numeric = 1, 
                            output_generator: Iterator[Union[str, int]] = None
                            ) -> Generator[Union[str, int], None, None]:
    
    """
    Returns a generator object which yields subsequent draws from a single dirichlet
    process.
    :param alpha: alpha parameter of the Dirichlet process
    :param output_generator: generator which yields unique symbols (likely hdphmm.state_generator)
    :return: generator object
    """

    # this should not be the case, since output_generator = self.state_generator
    if output_generator is None:

        output_generator = itertools.count(start=0, step=1)
    
    count = 0
    weights = {}
    
    while True:

        # if draw from unif(0,1) greater than ratio of count to count + alpha param, 
        # we generate a new state
        # Note: The larger alpha prior is, the more likely we will draw new states 
        if random.uniform(0, 1) > (count / (count + alpha)):

            val = next(output_generator)
            weights[val] = 1

        # otherwise, we will continue to reinforce existing states 
        else:

            val = np.random.choice(list(weights.keys()), 1, p=list(x / count for x in weights.values()))[0]
            weights[val] += 1
        
        count += 1
        
        # we yield the state we brew (be it brand new or one previously generated)
        # Note: I don't see how already removed states would be removed from 
        # and the originally initialised states would be icnorporated into the weights dict
        yield val


# used to ensure all hyperparameters have non-zero values
def max_dict(d: Dict[str, Numeric], eps: Numeric = 1e-8) -> Dict[str, Numeric]:
    
    return {k: max(float(v), eps) for k, v in d.items()}


def shrink_probabilities(d: Dict[Optional[str], Numeric], eps: Numeric = 1e-12) -> Dict[Optional[str], Numeric]:

    denom = sum(d.values()) + len(d) * eps
    
    return {k: (float(v) + eps) / denom for k, v in d.items()}
