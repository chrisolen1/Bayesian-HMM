#!/usr/bin/env python3
"""
Hierarchical Dirichlet Process Hidden Markov Model (HDPHMM).
The HDPHMM object collects a number of observed emission sequences, and estimates
latent states at every time point, along with a probability structure that ties latent
states to emissions. This structure involves

  + A starting probability, which dictates the probability that the first state
  in a latent seqeuence is equal to a given symbol. This has a hierarchical Dirichlet
  prior.
  + A transition probability, which dictates the probability that any given symbol in
  the latent sequence is followed by another given symbol. This shares the same
  hierarchical Dirichlet prior as the starting probabilities.
  + An emission probability, which dictates the probability that any given emission
  is observed conditional on the latent state at the same time point. This uses a
  Dirichlet prior.

Fitting HDPHMMs requires MCMC estimation. MCMC estimation is thus used to calculate the
posterior distribution for the above probabilities. In addition, we can use MAP
estimation (for example) to fix latent states, and facilitate further analysis of a
Chain.
"""
# Support typehinting.
from __future__ import annotations
from typing import Any, Union, Optional, Set, Dict, Iterable, List, Callable, Generator

import numpy as np
import random
import copy
import terminaltables
import tqdm
import functools
import multiprocessing
import string
from scipy import special, stats
from sympy.functions.combinatorial.numbers import stirling
from .chain import Chain
from .utils import label_generator, dirichlet_process_generator, shrink_probabilities
from warnings import catch_warnings

# Shorthand for numeric types.
Numeric = Union[int, float]

# Oft-used dictionary initializations with shorthands.
DictStrNum = Dict[Optional[str], Numeric]
InitDict = DictStrNum
DictStrDictStrNum = Dict[Optional[str], DictStrNum]
NestedInitDict = DictStrDictStrNum


class HDPHMM(object):
    """
    The Hierarchical Dirichlet Process Hidden Markov Model object. In fact, this is a
    sticky-HDPHMM, since we allow a biased self-transition probability.
    """

    def __init__(self,
                 emission_sequences: Iterable[List[Optional[str]]],
                 emissions=None,  # type: ignore
                 # emissions: Optional[Iterable[Union[str, int]]] = None # ???
                 sticky: bool = True,
                 priors: Dict[str, Callable[[], Any]] = None,) -> None:
        
        """
        Create a Hierarchical Dirichlet Process Hidden Markov Model object, which can
        (optionally) be sticky. The emission sequences must be provided, although all
        other parameters are initialised with reasonable default values. It is
        recommended to specify the `sticky` parameter, depending on whether you believe
        the HMM to have a high probability of self-transition.
        
        :param emission_sequences: iterable, containing the observed emission sequences.
        emission sequences can be different lengths, or zero length.
        
        :param emissions: set, optional. If not all emissions are guaranteed to be
        observed in the data, this can be used to force non-zero emission probabilities
        for unobserved emissions.
        
        :param sticky: bool, flag to indicate whether the HDPHMM is sticky or not.
        Sticky HDPHMMs have an additional value (kappa) added to the probability of self
        transition. It is recommended to set this depending on the knowledge of the
        problem at hand.
        
        :param priors: dict, containing priors for the model hyperparameters. Priors
        should be functions with zero arguments. The following priors are accepted:
          + alpha: prior distribution of the alpha parameter. Alpha
            parameter is the value used in the hierarchical Dirichlet prior for
            transitions and starting probabilities. Higher values of alpha keep rows of
            the transition matrix more similar to the beta parameters.
          + gamma: prior distribution of the gamma parameter. Gamma controls the
            strength of the uninformative prior in the starting and transition
            distributions. Hence, it impacts the likelihood of resampling unseen states
            when estimating beta coefficients. That is, higher values of gamma mean the
            HMM is more likely to explore new states when resampling.
          + alpha_emission: prior distribution of the alpha parameter for the
            emission prior distribution. Alpha controls how tightly the conditional
            emission distributions follow their hierarchical prior. Hence, higher values
            of alpha_emission mean more strength in the hierarchical prior.
          + gamma_emission: prior distribution of the gamma parameter for the
            emission prior distribution. Gamma controls the strength of the
            uninformative prior in the emission distribution. Hence, higher values of
            gamma mean more strength of belief in the prior.
          + kappa: prior distribution of the kappa parameter for the
            self-transition probability. Ignored if `sticky==False`. Kappa prior should
            have support in (0, 1) only. Higher values of kappa mean the chain is more
            likely to explore states with high self-transition probabilty.
        """
        
        # store (emission, latent var) chains generated from Chain module
        self.chains = [Chain(sequence) for sequence in emission_sequences]

        # sticky flag
        if type(sticky) is not bool:
            raise ValueError("`sticky` must be type bool")
        self.sticky = sticky

        # store hyperparameter priors
        self.priors = {"alpha": lambda: np.random.gamma(2, 2),
        "gamma": lambda: np.random.gamma(3, 3),
        "alpha_emission": lambda: np.random.gamma(2, 2),
        "gamma_emission": lambda: np.random.gamma(3, 3),
        "kappa": lambda: np.random.beta(1, 1)}

        ### RETURN TO THIS ###
        if priors is not None:
            self.priors.update(priors)

        if len(self.priors) > 5:
            raise ValueError("Unknown hyperparameter priors present")

        # set kappa equal to zero if HDP-HMM not sticky
        if not self.sticky:
            self.priors["kappa"] = lambda: 0

            if priors is not None and "kappa" in priors:
                raise ValueError("`sticky` is False, but kappa prior function given")

        # store param_name: param_value in hyperparams dict
        self.hyperparameters = {param: prior() for param, prior in self.priors.items()}

        # use internal properties to store counts, initialize to zero
        ### the number of times a state is the initial state of the sequence
        self.n_initial: InitDict 
        self.n_initial = {None: 0}
        ### the number times an emission <value> is categorized as a particular state <key>
        self.n_emission: NestedInitDict
        self.n_emission = {None: {None: 0}}
        ### number of transitions from state <key> to state <value>
        self.n_transition: NestedInitDict
        self.n_transition = {None: {None: 0}}

        # use internal properties to store current state for probabilities, initialize to one
        ### probability that each state is the initial state
        self.p_initial: InitDict
        self.p_initial = {None: 1}
        ### the probability that a particular emission <value> belongs to each state <key>
        self.p_emission: NestedInitDict
        self.p_emission = {None: {None: 1}}
        ### the probability of transitioning from state <key> to state <value>
        self.p_transition: NestedInitDict
        self.p_transition = {None: {None: 1}}

        # store derived parameters
        ### RETURN TO THIS ###
        self.auxiliary_transition_variables: NestedInitDict
        self.auxiliary_transition_variables = {None: {None: 0}}
        ### possibly beta outputs of Dirichlet stick-breaking ???
        self.beta_mixture_variables: InitDict
        self.beta_mixture_variables = {None: 1}
        ### RETURN TO THIS ###
        self.beta_emission: InitDict
        self.beta_emission = {None: 1}
        

        ### RETURN TO THIS ###
        # states & emissions
        # TODO: figure out emissions's type...
        if emissions is None:
            emissions = functools.reduce(  # type: ignore
                set.union, (set(c.emission_sequence) for c in self.chains), set())
        elif not isinstance(emissions, set):
            raise ValueError("emissions must be a set")
        
        self.emissions = emissions  # type: ignore
        self.states: Set[Optional[str]] = set()

        # generate non-repeating character labels for latent states
        self._label_generator = label_generator(string.ascii_lowercase)

        # keep flag to track initialisation
        self._initialised = False

    @property
    def initialised(self) -> bool:
        
        """
        Test whether a HDPHMM is initialised.
        :return: bool
        """
        
        return self._initialised

    ### RETURN TO THIS ###
    @initialised.setter
    def initialised(self, value: Any) -> None:
        
        if value:
            raise AssertionError("HDPHMM must be initialised through initialise method")
        elif not value:
            self._initialised = False
        else:
            raise ValueError("Initialised flag must be Boolean")

    @property
    def c(self) -> int:
        
        """
        Number of chains in the HMM.
        :return: int
        """
        
        return len(self.chains)

    @property
    def k(self) -> int:
        
        """
        Number of latent states in the HMM currently.
        :return: int
        """
        
        return len(self.states)

    @property
    def n(self) -> int:
        
        """
        Number of unique emissions. If `emissions` was specified when the HDPHMM was
        created, then this counts the number of elements in `emissions`. Otherwise,
        counts the number of observed emissions across all emission sequences.
        :return: int
        """
        
        return len(self.emissions)

    def tabulate(self) -> np.array:
        
        """
        Convert the latent and emission sequences for all chains into a single numpy
        array. Array contains an index which matches a chain's index in
        HDPHMM.chains, the current latent state, and the emission for all chains at
        all times.
        :return: numpy array with dimension (n, 3), where n is the length of all of the chains 
        e.g.: [[chain_#,state_name,emission],[chain_#,state_name,emission],[chain_#,state_name,emission]]
        """
        
        hmm_array = np.concatenate(
            tuple(np.concatenate((np.array([[n] * self.chains[n].T]).T, self.chains[n].tabulate()),axis=1,)
                  for n in range(self.c)),axis=0,)
        return hmm_array

    def __repr__(self) -> str:
        
        return "<bayesian_hmm.HDPHMM, size {C}>".format(C=self.c)

    def __str__(self, print_len: int = 15) -> str:
        
        fs = ("bayesian_hmm.HDPHMM," + " ({C} chains, {K} states, {N} emissions, {Ob} observations)")
        return fs.format(C=self.c, K=self.k, N=self.n, Ob=sum(c.T for c in self.chains))

    def state_generator(self, eps: Numeric = 1e-12) -> Generator[str, None, None]:
        
        """
        Create a new state for the HDPHMM, and update all parameters accordingly.
        This involves updating
          + The counts for the state
          + The auxiliary variables for the new state
          + The probabilities for the new state
          + The states captured by the HDPHMM
        :return: str, label of the new state
        """
        
        while True:
            
            ### generate unique label for new latent state 
            label = next(self._label_generator)

            
            """
            update counts with zeros (we assume update_counts() called later)
            state irrelevant for constant count (all zeros)
            """
            ### RETURN TO THIS ### do we want transitions to None and from None???
            ### number of times new state is initial state = 0
            self.n_initial[label] = 0
            ### n_transitions from new state to all other states = 0
            self.n_transition[label] = {s: 0 for s in self.states.union({label, None})}
            ### n_transitions from all other states to new state = 0
            for s in self.states:
                
                self.n_transition[s].update({label: 0})
            ### the number of emissions belong to the new state = 0
            self.n_emission[label] = {e: 0 for e in self.emissions}
            

            """
            update auxiliary transition variables
            """
            
            ### aux_transitions from new state to all other states = 1
            self.auxiliary_transition_variables[label] = {s2: 1 for s2 in list(self.states) + [label, None]}
            ### aux_transitions from all other states to new state = 1
            for s1 in self.states:
                
                self.auxiliary_transition_variables[s1][label] = 1
               

            """
            update beta_mixture_variables values via Dirichlet stick-breaking
            """
            
            ### Dirichlet stick-breaking: produce temp beta value using beta(1, gamma) from gamma prior
            temp_beta = np.random.beta(1, self.hyperparameters["gamma"])
            ### new beta for new state = temp_beta * old beta for None
            self.beta_mixture_variables[label] = temp_beta * self.beta_mixture_variables[None]
            ### new beta for None = (1 - temp_beta) * old beta for None
            self.beta_mixture_variables[None] = (1 - temp_beta) * self.beta_mixture_variables[None]


            ### RETURN TO THIS ###: Is this really how we determine initial state prob?
            """
            similarly, update initial state probabilities via Dirichlet stick-breaking
            """
            
            temp_p_initial = np.random.beta(1, self.hyperparameters["gamma"])
            self.p_initial[label] = temp_p_initial * self.p_initial[None]
            self.p_initial[None] = (1 - temp_p_initial) * self.p_initial[None]
            

            """
            update transition probabilities from new state to other states via Dirichlet process
            """
            
            ### RETURN TO THIS ### do we not want to factor in alpha's and kappas here?
            ### transition probabilities from new state to other states based on beta_mixture_variables (i.e. average transition probabilities)
            temp_p_transition = np.random.dirichlet([self.beta_mixture_variables[s] for s in list(self.states) + [label, None]]) 
            ### zipping state names and resulting transition probabilities
            p_transition_label = dict(zip(list(self.states) + [label, None], temp_p_transition))
            ### adding shrunk probabilities to p_transition under key of the new state
            self.p_transition[label] = shrink_probabilities(p_transition_label, eps)
            

            """
            update transitions from other states into new state via Dirichlet stick-breaking
            """
            
            ### iterate through all states, notably excluding the new state
            for state in self.states.union({None}):
                
                ### draw from gamma prior
                temp_p_transition = np.random.beta(1, self.hyperparameters["gamma"])
                ### new p_transition from other state to new state = temp_beta * old p_transition from other state to None 
                self.p_transition[state][label] = (self.p_transition[state][None] * temp_p_transition)
                ### new p_transition from other state to None = (1 - temp_beta) * old p_transition from other state to None 
                self.p_transition[state][None] = self.p_transition[state][None] * (1 - temp_p_transition)
            

            """
            update emission probabilities
            """
            
            ### RETURN TO THIS ### is this really how we determine emissions probabilities for a state??? or
            ### Dirichlet process for creating emissions probabilities for each possible emission for each state 
            temp_p_emission = np.random.dirichlet([self.hyperparameters["alpha"] * self.beta_emission[e] for e in self.emissions])
            ### for the new state, zip resulting emissions probability with the emission itself
            self.p_emission[label] = dict(zip(self.emissions, temp_p_emission))

            # save generated label to state dict
            self.states = self.states.union({label})

            yield label

    def initialise(self, k: int = 20) -> None:
        
        """
        Initialise the HDPHMM. This involves:
          + Choosing starting values for all hyperparameters
          + Initialising all chains (see chain.initialise for further info)
          + Initialising priors for probabilities (i.e. the Hierarchical priors)
          + Updating all counts
          + Sampling latent states and auxiliary beam variables
        Typically called directly from a HDPHMM object.
        :param k: number of hidden states
        :return: None
        """
        
        # create as many states as specified by k 
        states = [next(self._label_generator) for _ in range(k)]
        self.states = set(states)

        ### RETURN TO THIS ### already did this above???
        # create dict of hyperparams drawn from priors
        #self.hyperparameters = {param: prior() for param, prior in self.priors.items()}

        # initialise chains: creates randomly initialised hidden state sequence for each emission sequence/chain
        for c in self.chains:
            c.initialise(states)

        # initialise hierarchical priors
        
        ### order L weak limit approximation to the Dirichlet process mixture where L = self.k in this case 
        temp_beta = sorted(np.random.dirichlet([self.hyperparameters["gamma"] / (self.k + 1)] * (self.k + 1)), reverse=True)
        ### create dictionary of beta_mixture_variables with corresponding state names
        beta_mixture_variables = dict(zip(list(self.states) + [None], temp_beta))
        ### and shrink probabilities
        self.beta_mixture_variables = shrink_probabilities(beta_mixture_variables)
        ### set aux_transition from each state to each other equal to 1
        self.auxiliary_transition_variables = {s1: {s2: 1 for s2 in self.states.union({None})} for s1 in self.states.union({None})}

        # update counts before resampling
        self.update_counts()

        # resample remaining hyperparameters
        self.resample_beta_mixture_variables()
        self.resample_beta_emission()
        self.resample_p_initial()
        self.resample_p_transition()
        self.resample_p_emission()

        # update initialised flag
        self._initialised = True

    def remove_unused_states(self):

        """
        Remove defunct states from the internal set of states, and merge all parameters
        associated with these states back into the 'None' values.
        """

        # assign copy of current states to states_prev
        states_prev = copy.copy(self.states)
        # get only states assigned as latent variables in current iteration and assign to states_next
        states_next = set(sorted(functools.reduce(set.union, (set(c.latent_sequence) for c in self.chains), set())))
        # single out unused states for removal
        states_removed = states_prev - states_next

        # remove relevant key:values from prop dicts and add prob to aggregate None state
        for state in states_removed:

            self.beta_mixture_variables[None] += self.beta_mixture_variables.pop(state)
            self.p_initial[None] += self.p_initial.pop(state)
            for s1 in states_next.union({None}):
                self.p_transition[s1][None] += self.p_transition[s1].pop(state)

            # remove transition vector entirely
            del self.p_transition[state]

        # update internal state tracking
        self.states = states_next

    def update_counts(self):
        
        """
        Update counts required for subsequently resampling probabilities. These counts are used
        to sample from the posterior distribution for probabilities. This function
        should be called after any latent state is changed, including after resampling.
        :return: None
        """
        
        # check that all chains are initialised
        if any(not chain.initialised_flag for chain in self.chains):
            raise AssertionError("Chains must be initialised before calculating fit parameters")

        # transition count for non-oracle transitions
        
        # set n_initial, n_emission, and n_transition objects (NOT their corresponding class objects) to 0 for each state 
        n_initial = {s: 0 for s in self.states.union({None})}
        n_emission = {s: {e: 0 for e in self.emissions} for s in self.states.union({None})}
        n_transition = {s1: {s2: 0 for s2 in self.states.union({None})} for s1 in self.states.union({None})}

        # increment all relevant hyperparameters while looping over sequence/chain
        for chain in self.chains:
            
            # increment n_initial state if said state is the initial state of the sequence
            n_initial[chain.latent_sequence[0]] += 1

            # increment n_emissions only for final state--emission pair
            n_emission[chain.latent_sequence[chain.T - 1]][chain.emission_sequence[chain.T - 1]] += 1

            # increment all n_transitions and n_emissions within chain
            for t in range(chain.T - 1):
                
                # within chain emissions
                n_emission[chain.latent_sequence[t]][chain.emission_sequence[t]] += 1
                # within chain transitions
                n_transition[chain.latent_sequence[t]][chain.latent_sequence[t + 1]] += 1

        # store new count hyperparameters as class objects
        self.n_initial = n_initial
        self.n_emission = n_emission
        self.n_transition = n_transition

    @staticmethod
    def exact_resample_aux_transition_counts(alpha,
                                            beta,
                                            n_jk,
                                            use_approximation=True,
                                            state_pair):
        
        """
        Use a resampling approach that estimates probabilities for all auxiliary
        transition parameters. This avoids the slowdown in convergence caused by
        Metropolis Hastings rejections, but is more computationally costly.
        :param alpha: parameter drawn from alpha prior
        :param beta: beta param corresponding to the second state in the key-value state pairing
        :param n_jk: tthe total number of customers in restaurant j (corresponding to state1)
                    eating dish k (corresponding to state 2)
        :param use_approximation: boolean
        :return:
        """
        
        # probability threshold required to cross
        required_prob = np.random.uniform(0, 1)
        # number of unique tables in the restaurant, meaning the number of unique 
        # latent states that transitioned to the latent state in question 
        # ultimately, we trying to determine p(m_jk = m | n_jk, alpha, beta, kappa)
        # and trying out a bunch of different m's until we reach a probability 
        # threshold that we're happy with 
        m = 0
        cumulative_prob = 0
        # perform this computation just once now:
        ab_k = alpha * beta
        # euler constant if approximation necessary
        euler_constant = 0.5772156649
        # ordered pair of states (i.e. state1 transitions to state2)
        state1, state2 = state_pair
        # indicator variable for self-transition
        if state1 == state2:
            delta = 1
        else:
            delta = 0

        # use precise probabilities
        if not use_approximation:
            
            try:
                
                log_gamma_constant = np.log(special.gamma(ab_k + (self.hyperparameters["kappa"]*delta))) 
                                    - np.log(special.gamma(ab_k + (self.hyperparameters["kappa"]*delta) + n_jk))
                
                # continue while cumulative_prob < requirement AND 
                # m (# of tables in rest) < n (number of customers in rest)
                while cumulative_prob < required_prob and m < n:
                    
                    # adding log-transformed terms
                    density = log_gamma_constant 
                            + np.log(stirling(n_jk, m, kind=1)) 
                            + (m * np.log(ab_k + (self.hyperparameters["kappa"]*delta)))
                    # adding resulting p(mjk = m | z, m-jk, b) to cumulative_prob
                    cumulative_prob += np.exp(density)
                    # increment m by one
                    m += 1
            
            # after one failure use only the approximation
            except (RecursionError, OverflowError):
                
                # correct for previous failed instance
                m -= 1
        
        """
        LOOKING FOR DERIVATION 
        # use approximate probabilities
        # problems may occur with stirling recursion (e.g. large n & m)
        # magic number is the Euler constant
        # approximation derived in documentation
        while cumulative_prob < cumulative_prob and m < n:
            
            density = m 
                + ((m + ab_k - 0.5) * np.log(ab_k)) 
                + ((m - 1) * np.log(euler_constant))
                + np.log(n_jk - 1)
                - ((m - 0.5) * np.log(m)) 
                - (ab_k * np.log(ab_k + n_jk)) 
                - ab_k
            
            cumulative_prob += np.exp(density)
            # increment m by one
            m += 1
        """
        # breaks out of loop after m is sufficiently large
        return max(m, 1)
    

    @staticmethod
    def mh_resample_aux_transition_counts(alpha,
                                        beta,
                                        n_jk,
                                        m_curr,
                                        use_approximation=True,
                                        state_pair):
        """
        Use a Metropolos Hastings resampling approach that often rejects the proposed
        value. This can cause the convergence to slow down (as the values are less
        dynamic) but speeds up the computation.
        :param alpha: parameter drawn from alpha prior
        :param beta: beta param corresponding to the second state in the key-value state pairing
        :param n_jk: the total number of customers in restaurant j (corresponding to state1)
                    eating dish k (corresponding to state 2)
        :param m_curr: takes auxiliary_transition_variables as an argument
        :param use_approximation:
        :return:
        """

        # propose new m
        n_jk = max(n_jk, 1)
        m_proposed = random.choice(range(1, n_jk + 1))
        # perform this computation just once now:
        ab_k = alpha * beta
        # euler constant if approximation necessary
        euler_constant = 0.5772156649
        # ordered pair of states (i.e. state1 transitions to state2)
        state1, state2 = state_pair
        
        if m_curr > n_jk:
            
            return m_proposed

        # find relative probabilities
        if not use_approximation:

            p_curr = float(stirling(n_jk, m_curr, kind=1)) * (ab_k ** m_curr)
            p_proposed = float(stirling(n_jk, m_proposed, kind=1)) * (ab_k ** m_proposed)
            logp_diff = np.log(p_proposed) - np.log(p_curr)
    """
        DONT UNDERSTAND THE MATH HERE 
        else:

            logp_diff = ((m_proposed - 0.5) * np.log(m_proposed)) 
                    - (m_curr - 0.5) * np.log(m_curr)
                    + (m_proposed - m_curr) * np.log(ab_k * np.exp(1))
                    + (m_proposed - m_curr) * np.log(euler_constant) 
                    + np.log(n_jk - 1)

    """

        # use MH variable to decide whether to accept m_proposed
        with catch_warnings(record=True) as caught_warnings:
            p_accept = min(1, np.exp(logp_diff))
            p_accept = bool(np.random.binomial(n=1, p=p_accept))  # convert to boolean
            if caught_warnings:
                p_accept = True

        return m_proposed if p_accept else m_curr

    @staticmethod
    def _resample_auxiliary_transition_atom(state_pair,
                                            alpha,
                                            beta,
                                            n_initial,
                                            n_transition,
                                            auxiliary_transition_variables,
                                            resample_type="exact",
                                            use_approximation=True):
        
        """
        Resampling the auxiliary transition atoms should be performed before resampling
        the transition beta values. This is the static method, created to allow for
        parallelised resampling.
        :param state_pair: ordered pair of states
        :param alpha: param drawn from the alpha prior
        :param beta: beta_mixture_variables from Dirichlet stick-breaking
        :param n_initial: n_initial class object
        :param n_transition: n_transition class object
        :param auxiliary_transition_variables: auxiliary_transition_variables class object
        :param resample_type: either mh or complete
        :param use_approximation: boolean
        :return:
        """

        # ordered pair of states (i.e. state1 transitions to state2)
        state1, state2 = state_pair

        # apply resampling if 'mh' == True
        if resample_type == "mh":
            
        	# we focus on instances where state2 is the initial state OR
        	# when transitioning from state1 to state2
            # i.e. when state2 is the dish served per CRF
            return HDPHMM.mh_resample_aux_transition_counts(alpha,
                                                            beta[state2],
                                                            n_initial[state2] + n_transition[state1][state2],
                                                            auxiliary_transition_variables[state1][state2],
                                                            use_approximation,
                                                            state_pair)
        # apply resampling if 'complete' == True
        elif resample_type == "exact":
            
            # we focus on instances where state2 is the initial state OR
        	# when transitioning from state1 to state2
            # i.e. when state2 is the dish served per CRF
        	# Note: This approach does not use aux vars
            return HDPHMM.exact_resample_aux_transition_counts(alpha,
                                                                beta[state2],
                                                                n_initial[state2] + n_transition[state1][state2],
                                                                use_approximation,
                                                                state_pair)
        
        else:
            raise ValueError("resample_type must be either mh or complete")

    def crp_resample_aux_transition_counts(self):
        
        """
        For large n_jk, it is often more efficient to sample m_jk by simulating table 
        assignments of the Chinese restaurant, rather than having to compute a large
        array of Stirling numbers. Having the state index assignments z1:T effectively
        partitions the data (customers) into both restaurants and dishes, though the table 
        assignments are unknown since multiple tables can be served the same dish. Thus,
        sampling m_jk is in effect requivalent to sampling table assignments for each 
        customer AFTER knowing the dish assignment
        :return:
        """
        
        # establish empty dict for counting the number of separate tables per state in each restaurant
        # eventually will take the form {restaurant: dish: table_number: num_customers}
        tables = {}
        # store alpha
        alpha = self.hyperparameters['alpha']
        kappa = self.hyperparameters['kappa']
        
        # iterate through chains one at a time 
        for chain in self.chains:

            # if initial state, it automatically get its own table with probabiity 1
            init_dish = chain.latent_sequence[0]
            tables.update({init_dish: {init_dish: {1: 1}}})

            # go state by state in the latent sequence
            for t in range(1, len(chain.latent_sequence)):

                # get the current and previous state names
                prev_dish = chain.latent_sequence[t-1]
                current_dish = chain.latent_sequence[t]

                # if no one has ordered the current dish in the current restaurant
                if current_dish not in tables[prev_dish].keys():

                    # it automatically gets a table
                    tables.update({prev_dish: {current_dish: {1: 1}}})

                else:

                    # get the index of the table in the current restaurant (i.e. prev_dish)
                    # which most recently chose the current_dish
                    d = tables[prev_dish][current_dish]
                    od = collections.OrderedDict(sorted(d.items()))
                    latest_table_index = list(od.items())[-1][0]
                    # and then find the number of customers sitting at that tables
                    n_jt = tables[prev_dish][current_dish][latest_table_index]
                    
                    # get beta and kappa (if self-transition) for the current dish
                    beta_k = self.beta_mixture_variables[current_dish]
                    if prev_dish == current_dish:
                        delta = 1
                    else:
                        delta = 0
                    new_table_weight = alpha*beta_k + delta*kappa

                    # make random draw to decide whether to sit at existing or new tables
                    result = random.choices(('old_table','new_table'), weights=(n_jt, new_table_weight))[0]

                    # if we stay at the existing table
                    if result == 'old_table':

                        # add one customer the table that last ordered the current_dish
                        # in the restaurant we're currently in 
                        tables[prev_dish][current_dish][latest_table_index] += 1

                    else:

                        # else, create a new table in the restaurant for our current dish
                        # and add a customer 
                        new_latest_index = latest_table_index + 1
                        tables[prev_dish][current_dish][new_latest_index] = 1

            # now time to update the auxiliary_transition_variable class variable
            # for each restaurant key in aux_trans_var
            for restaurant in self.auxiliary_transition_variables:

                # for each dish key in each restaurant
                for dish in self.auxiliary_transition_variables[restaurant]:

                    # set the aux_tran_var restaurant-dish pair equal to the number
                    # of distinct tables in that restaurant serving that dish (i.e. m_jk)
                    self.auxiliary_transition_variables[restaurant][dish] = len(tables[restaurant][dish].keys())

    def gem_resample_aux_transition_counts(self):

        """
        For large n_jk, it is often more efficient to sample m_jk by simulating table 
        assignments of the Chinese restaurant, rather than having to compute a large
        array of Stirling numbers. The form used for the CRP above implies that a 
        customer's table assignment conditioned on a dish assignment k follows a 
        Dirichlet process with concentration parameter alpha*beta_k + kappa•delta(k,j)
        such that  t_ji ~ GEM(alpha*beta_k + kappa•delta(k,j))

        :return:
        """

        alpha = self.hyperparameters['alpha']
        kappa = self.hyperparameters['kappa']

        # initial a separate probability stick for each state (i.e. dish)
        # where None currently occupies the totality of the density 
        dish_sticks = {dish: {None: 1} for dish in self.states}

        # iterate through chains one at a time 
        for chain in self.chains:

            # go state by state in the latent sequence
            for t in range(len(chain.latent_sequence)):

                # self-transition is not possible for initial state in the chain
                if t == 0:

                    dish = chain.latent_sequence[t]

                        # if this is the first stick-break for this particular dish
                        # we'll automatically break a pice off of None with probability 1
                        if len(dish_sticks[dish].keys()) == 1:

                            # note: we leave out kappa given self-transition not possible
                            temp_p = np.random.beta(1, alpha * self.beta_mixture_variables[dish])
                            # create new table for this dish with segment length dish_sticks[dish][None] * temp_p
                            dish_sticks[dish][1] = dish_sticks[dish][None] * temp_p
                            # remainder of the segment length goes to None
                            dish_sticks[dish][None] = dish_sticks[dish][None] * (1 - temp_p)

                        else:

                            # pull the penultimate segment index for the current_dish
                            # (i.e. up until None)
                            d = dish_sticks[current_dish]
                            latest_table_index = list(d.items())[-1][0]
                            # sum up the segmented density up, excluding None density, which is
                            # the first index
                            segmented_density = sum(list(dish_sticks[current_dish].values())[1:])
                            # make a uniform draw from 0 to 1
                            draw = random.uniform(0,1)

                            # if draw puts us passed the segmented density of the stick
                            if draw > segmented_density:

                                # break off a new segment
                                # note: we leave out kappa given self-transition not possible
                                temp_p = np.random.beta(1, alpha * self.beta_mixture_variables[dish])
                                # create new table for this dish with segment length dish_sticks[dish][None] * temp_p
                                dish_sticks[dish][latest_table_index+1] = dish_sticks[dish][None] * temp_p
                                # remainder of the segment length goes to None
                                dish_sticks[dish][None] = dish_sticks[dish][None] * (1 - temp_p)


                # this is where we started to incorporate kappa
                else:

                    previous_dish = chain.latent_sequence[t-1]
                    current_dish = chain.latent_sequence[t]

                    if current_dish == previous_dish:
                        delta = 1
                    else:
                        delta = 

                    # pull the penultimate segment index for the current_dish
                            # (i.e. up until None)
                            d = dish_sticks[current_dish]
                            latest_table_index = list(d.items())[-1][0]
                            # sum up the segmented density up, excluding None density, which is
                            # the first index
                            segmented_density = sum(list(dish_sticks[current_dish].values())[1:])
                            # make a uniform draw from 0 to 1
                            draw = random.uniform(0,1)

                            # if draw puts us passed the segmented density of the stick
                            if draw > segmented_density:

                                # break off a new segment
                                # note: we leave out kappa given self-transition not possible
                                temp_p = np.random.beta(1, (alpha * self.beta_mixture_variables[dish]) + (kappa * delta))
                                # create new table for this dish with segment length dish_sticks[dish][None] * temp_p
                                dish_sticks[dish][latest_table_index+1] = dish_sticks[dish][None] * temp_p
                                # remainder of the segment length goes to None
                                dish_sticks[dish][None] = dish_sticks[dish][None] * (1 - temp_p)
 
        # make resulting dish_sticks dict a class variable
        self.dish_sticks = dish_sticks

    ### transition probabilities from new state to other states based on beta_mixture_variables (i.e. average transition probabilities)
            temp_p_transition = np.random.dirichlet([self.beta_mixture_variables[s] for s in list(self.states) + [label, None]]) 
            ### zipping state names and resulting transition probabilities
            p_transition_label = dict(zip(list(self.states) + [label, None], temp_p_transition))
            ### adding shrunk probabilities to p_transition under key of the new state
            self.p_transition[label] = shrink_probabilities(p_transition_label, eps)

        for state in self.states.union({None}):
                
                ### draw from gamma prior
                temp_p_transition = np.random.beta(1, self.hyperparameters["gamma"])
                ### new p_transition from other state to new state = temp_beta * old p_transition from other state to None 
                self.p_transition[state][label] = (self.p_transition[state][None] * temp_p_transition)
                ### new p_transition from other state to None = (1 - temp_beta) * old p_transition from other state to None 
                self.p_transition[state][None] = self.p_transition[state][None] * (1 - temp_p_transition)    

    # TODO: decide whether to use either MH resampling or approximation sampling and
    # remove the alternative, unnecessary complexity in code
    def _resample_auxiliary_transition_variables(self, 
                                                 ncores=1, 
                                                 resample_type="exact", 
                                                 use_approximation=True):
        
        # non-multithreaded process uses typical list comprehension
        if ncores < 2 and resample_type == "mh" or resample_type == "exact":
            
            # iterate over all possible markov transition permutations of states and send then them through
            # the _resample_auxiliary_transition_atom function
            # updates aux_trans_var m_jk based on estimate
            self.auxiliary_transition_variables = {s1: {s2: HDPHMM._resample_auxiliary_transition_atom((s1, s2),
                													alpha=self.hyperparameters["alpha"],
                													beta=self.beta_mixture_variables,
                													n_initial=self.n_initial,
                													n_transition=self.n_transition,
                													auxiliary_transition_variables=self.auxiliary_transition_variables,
                													resample_type=resample_type,
                													use_approximation=use_approximation) 
            										for s2 in self.states} for s1 in self.states}

        # parallel process uses anonymous functions and mapping
        elif ncores > 2 and resample_type == "mh" or resample_type == "exact":

            # specify ordering of states
            state_pairs = [(s1, s2) for s1 in self.states for s2 in self.states]

            # parallel process resamples
            resample_partial = functools.partial(HDPHMM._resample_auxiliary_transition_atom,
                							alpha=self.hyperparameters["alpha"],
                							beta=self.beta_mixture_variables,
                							n_initial=self.n_initial,
                							n_transition=self.n_transition,
                							auxiliary_transition_variables=self.auxiliary_transition_variables,
                							resample_type=resample_type,
                							use_approximation=use_approximation)

            pool = multiprocessing.Pool(processes=ncores)
            auxiliary_transition_variables = pool.map(resample_partial, state_pairs)
            pool.close()

            # store as dictionary
            for pair_n in range(len(state_pairs)):
                state1, state2 = state_pairs[pair_n]
                self.auxiliary_transition_variables[state1][state2] = auxiliary_transition_variables[pair_n]

        elif resample_type == "crp":

            self.crp_resample_aux_transition_counts()

        elif resample_type == "gem":

            self.gem_resample_aux_transition_counts()            



    def _get_beta_mixture_variables_metaparameters(self):

        """
        Calculate parameters for the Dirichlet posterior of the transition beta
        variables (with infinite states aggregated into 'None' state)
        :return: dict, with a key for each state and None, and values equal to parameter
        values
        """

        # beta ~ Dir(m_.1, m_.2, ... , m_.K, gamma)
        if self.resample_type not "gem":

            dir_posterior_params_beta_mixture_variables = {s2: sum(self.auxiliary_transition_variables[s1][s2] for s1 in self.states)
                for s2 in self.states}

        else:

            # if we use gem resampling method, we just take the length (i.e. number of partitions)
            # of the dish_sticks dictionaries 
            dir_posterior_params_beta_mixture_variables = {s1: len(list(self.dish_sticks[s1].keys()))-1 for s1 in self.states}
            
        
        # posterior param for None is always gamma per stick-breaking process 
        dir_posterior_params_beta_mixture_variables[None] = self.hyperparameters["gamma"]

        return dir_posterior_params_beta_mixture_variables

    def resample_beta_mixture_variables(self, ncores=1, auxiliary_resample_type="mh", use_approximation=True, eps=1e-12):
        
        """
        Resample the beta values used to calculate the starting and transition
        probabilities.
        :param ncores: int. Number of cores to use in multithreading. Values below 2
        mean the resampling step is not parallelised.
        :param auxiliary_resample_type: either "mh" or "complete". Impacts the way
        in which the auxiliary transition variables are estimated.
        :param use_approximation: bool, flag to indicate whether an approximate
        resampling should occur. ignored if `auxiliary_resample_type` is "mh"
        :param eps: shrinkage parameter to avoid rounding error.
        :return: None
        """
        
        # auxiliary variables must be resampled prior to resampling beta variables
        self._resample_auxiliary_transition_variables(ncores=ncores,
                                                      resample_type=auxiliary_resample_type,
                                                      use_approximation=use_approximation)

        # get dir posterior params for beta_mixture_variables now that we've resampled m_jk's
        dir_posterior_params_beta_mixture_variables = self._get_beta_mixture_variables_metaparameters()
        # resample from Dirichlet posterior and overwrite beta_mixture_variables class variable with new sample
        beta_mixture_variables = dict(zip(list(dir_posterior_params_beta_mixture_variables.keys()), 
            np.random.dirichlet(list(dir_posterior_params_beta_mixture_variables.values())).tolist()))
        self.beta_mixture_variables = shrink_probabilities(beta_mixture_variables, eps)

    def calculate_beta_mixture_variables_loglikelihood(self):
        
        # get dir posterior params for beta_mixture_variables
        dir_posterior_params_beta_mixture_variables = self._get_beta_mixture_variables_metaparameters()
        # first argument is the quantiles of new beta_mixture_variables sample and 
        # second argument is Dirichlet posterior params
        new_beta_mixture_variables_prob_density = np.log(stats.dirichlet.pdf([self.beta_mixture_variables[s] for s in beta_posterior_params.keys()],
                [dir_posterior_params_beta_mixture_variables[s] for s in beta_posterior_params.keys()]))

        return new_beta_mixture_variables_prob_density


    def _get_p_initial_metaparameters(self):
        
        """
        Calculate parameters for the Dirichlet posterior of the p_initial variables
        (with infinite states aggregated into 'None' state)
        :return: dict, with a key for each initial state probability, and values equal to parameter
        values
        """

        # parameters for dirichlet posterior for initial probabilities equal to the
        # counts n_initial for each state plus the product of the corresponding beta value for 
        # the state and the alpha parameter
        dir_posterior_params_initial = {s: self.n_initial[s]
            + (self.hyperparameters["alpha"] * self.beta_mixture_variables[s]) for s in self.states}
        dir_posterior_params_initial[None] = (self.hyperparameters["alpha"] * self.beta_mixture_variables[None])
        
        return dir_posterior_params_initial

    def resample_p_initial(self, eps=1e-12):

        """
        Resample the starting probabilities. Performed as a sample from the posterior
        distribution, which is a Dirichlet with pseudocounts and actual counts combined.
        :param eps: minimum expected value.
        :return: None.
        """

        dir_posterior_params_initial = self._get_p_initial_metaparameters()
        # resample from Dirichlet posterior and overwrite p_initial class variable
        p_initial = dict(zip(list(dir_posterior_params_initial.keys()), 
            np.random.dirichlet(list(dir_posterior_params_initial.values())).tolist()))
        self.p_initial = shrink_probabilities(p_initial, eps)

    def calculate_p_initial_loglikelihood(self):
        
        dir_posterior_params_initial = self._get_p_initial_metaparameters()
        # first argument is the quantiles of new p_inital sample and 
        # second argument is Dirichlet posterior params
        new_initial_prob_density = np.log(stats.dirichlet.pdf([self.p_initial[s] for s in dir_posterior_params_initial.keys()],
                [dir_posterior_params_initial[s] for s in dir_posterior_params_initial.keys()]))
        
        return ll_initial

    def _get_p_transition_metaparameters(self, state):
        
        if self.sticky:
            
            # counting number of times we ate at restaurant state 
            # and ordered dish s2, meaning we transitioned from 
            # state to s2
            dir_posterior_params_pi_k = {s2: self.n_transition[state][s2]
                + (self.hyperparameters["alpha"] * (1 - self.hyperparameters["kappa"]) * self.beta_mixture_variables[s2])
                for s2 in self.states}

            dir_posterior_params_pi_k[None] = (self.hyperparameters["alpha"]
                * (1 - self.hyperparameters["kappa"])
                * self.beta_mixture_variables[None])

            # adding kappa back into the self transition; this seems like an odd way to do this 
            dir_posterior_params_pi_k[state] += (self.hyperparameters["alpha"] * self.hyperparameters["kappa"])
        
        else:
            
            dir_posterior_params_pi_k = {s2: self.n_transition[state][s2]
                + self.hyperparameters["alpha"] * self.beta_mixture_variables[s2]
                for s2 in self.states}

            dir_posterior_params_pi_k[None] = self.hyperparameters["alpha"] * self.beta_mixture_variables[None]

        return dir_posterior_params_pi_k

    def resample_p_transition(self, eps=1e-12):
        
        """
        Resample the transition probabilities from the current beta values and kappa
        value, if the chain is sticky.
        :param eps: minimum expected value passed to Dirichlet sampling step
        :return: None
        """

        # empty current transition values
        self.p_transition = {}

        # for each state (restaurant), sample from dir posterior params corresponding
        # to transitions to other states
        for state in self.states:
            dir_posterior_params_pi_k = self._get_p_transition_metaparameters(state)
            p_transition_state = dict(zip(list(dir_posterior_params_pi_k.keys()), 
                np.random.dirichlet(list(dir_posterior_params_pi_k.values())).tolist()))
            self.p_transition[state] = shrink_probabilities(p_transition_state, eps)

        # add transition probabilities from unseen states using beta_mixture_variables a dir posterior params
        # note: no stickiness update because these are aggregated states
        params = {k: self.hyperparameters["alpha"] * v for k, v in self.beta_mixture_variables.items()}
        p_transition_none = dict(zip(list(params.keys()), 
            np.random.dirichlet(list(params.values())).tolist()))
        self.p_transition[None] = shrink_probabilities(p_transition_none, eps)

    def calculate_p_transition_loglikelihood(self):
        
        """
        Note: this calculates the likelihood over all entries in the transition matrix.
        If chains have been resampled (this is the case during MCMC sampling, for
        example), then there may be entries in the transition matrix that no longer
        correspond to actual states.
        :return:
        """

        new_trans_prob_density = 0
        states = self.p_transition.keys()

        for state in states:

            dir_posterior_params_pi_k = self._get_p_transition_metaparameters(state)
            # add the log prob densities for transition probabilities for each restaurant 
            # first argument is the quantiles of new p_transition sample for each restaurant and 
            # second argument is Dirichlet posterior params 
            new_trans_prob_density += np.log(stats.dirichlet.pdf([self.p_transition[state][s] for s in states],
                    [dir_posterior_params_pi_k[s] for s in states]))

        # get probability for aggregate, None state
        params = {k: self.hyperparameters["alpha"] * v for k, v in self.beta_mixture_variables.items()}
        new_trans_prob_density += np.log(stats.dirichlet.pdf([self.p_transition[None][s] for s in states], 
            [params[s] for s in states]))

        return new_trans_prob_density

    def _get_beta_emission_metaparameters(self):
        
        """
        Calculate parameters for the Dirichlet posterior of the emission beta variables,
        i.e. probability of the emissions in general
        (with infinite states aggregated into 'None' state)
        :return: dict, with a key for each emission, and values equal to parameter
        values
        """

        # aggregate counts for each emission weighted by hyperparam
        dir_posterior_params_emissions = {e: sum(self.n_emission[s][e] for s in self.states)
            + self.hyperparameters["gamma_emission"] / self.n for e in self.emissions}

        return dir_posterior_params_emissions

    def resample_beta_emission(self, eps=1e-12):
        
        """
        Resample the beta values used to calculate the emission probabilties.
        :param eps: Minimum value for expected value before resampling.
        :return: None.
        """
        
        # get dir posterior params for emissions
        dir_posterior_params_emissions = self._get_beta_emission_metaparameters()
        # resample from Dirichlet posterior and overwrite beta_emission class variable with new sample
        beta_emission = dict(zip(list(dir_posterior_params_emissions.keys()), 
            np.random.dirichlet(list(dir_posterior_params_emissions.values())).tolist()))
        self.beta_emission = shrink_probabilities(beta_emission, eps)

    def calculate_beta_emission_loglikelihood(self):
        
        # get dir posterior params for emissions
        dir_posterior_params_emissions = self._get_beta_emission_metaparameters()
        # first argument is the quantiles of new beta_emission sample and 
        # second argument is Dirichlet posterior params
        new_emissions_prob_density = np.log(stats.dirichlet.pdf([self.beta_emission[e] for e in self.emissions],
                [dir_posterior_params_emissions[e] for e in self.emissions]))
        
        return new_emissions_prob_density

    def _get_p_emission_metaparameters(self, state):

        """
        Calculate parameters for the Dirichlet posterior of the p_emissions variables,
        i.e. probability of an emission given a state
        (with infinite states aggregated into 'None' state)
        :return: dict, with a key for each emission, and values equal to parameter
        values
        """
        
        # for each state, get the number of emissions weighted by beta_emission and alpha param
        dir_posterior_params_emissions_per_state = {e: self.n_emission[state][e] 
            + self.hyperparameters["alpha_emission"] * self.beta_emission[e]
            for e in self.emissions}
        
        return dir_posterior_params_emissions_per_state

    def resample_p_emission(self, eps=1e-12):
        
        """
        resample emission parameters from emission priors and counts.
        :param eps: minimum expected value passed to Dirichlet distribution
        :return: None
        """

        # for each state, sample from the dirichlet posterior params corresonding to 
        # the emissions probabilities given that state
        for state in self.states:
            dir_posterior_params_emissions_per_state = self._get_p_emission_metaparameters(state)
            p_emission_state = dict(zip(list(dir_posterior_params_emissions_per_state.keys()), 
                np.random.dirichlet(list(dir_posterior_params_emissions_per_state.values())).tolist()))
            self.p_emission[state] = shrink_probabilities(p_emission_state, eps)

        # add emission probabilities from unseen states
        params = {k: self.hyperparameters["alpha_emission"] * v for k, v in self.beta_emission.items()}
        p_emission_none = dict(zip(list(params.keys()), np.random.dirichlet(list(params.values())).tolist()))
        self.p_emission[None] = shrink_probabilities(p_emission_none, eps)

    def calculate_p_emission_loglikelihood(self):
        
        new_emission_per_state_prob_density = 0

        # add the log prob densities for emissions given each state
        # first argument is the quantiles of new p_emissions sample given each state and 
        # second argument is Dirichlet posterior params for emissions probabilities given a state
        for state in self.states:

            dir_posterior_params_emissions_per_state = self._get_p_emission_metaparameters(state)
            new_emission_per_state_prob_density += np.log(stats.dirichlet.pdf([self.p_emission[state][e] for e in self.emissions],
                    [dir_posterior_params_emissions_per_state[e] for e in self.emissions]))

        # get probability for aggregate state
        params = {k: self.hyperparameters["alpha_emission"] * v for k, v in self.beta_emission.items()}
        new_emission_per_state_prob_density += np.log(
            stats.dirichlet.pdf([self.p_emission[None][e] for e in self.emissions],
                [params[e] for e in self.emissions]))

        return new_emission_per_state_prob_density

    def print_fit_parameters(self):

        """
        Prints a copy of the current state counts.
        Used for convenient checking in a command line environment.
        For dictionaries containing the raw values, use the `n_*` attributes.
        :return:
        """

        # create copies to avoid editing
        n_initial = copy.deepcopy(self.n_initial)
        n_emission = copy.deepcopy(self.n_emission)
        n_transition = copy.deepcopy(self.n_transition)

        # make nested lists for clean printing
        initial = [[str(s)] + [str(n_initial[s])] for s in self.states]
        initial.insert(0, ["S_i", "Y_0"])
        emissions = [[str(s)] + [str(n_emission[s][e]) for e in self.emissions]
            for s in self.states]

        emissions.insert(0, ["S_i \\ E_i"] + list(map(str, self.emissions)))
        transitions = [[str(s1)] + [str(n_transition[s1][s2]) for s2 in self.states]
            for s1 in self.states]
        transitions.insert(0, ["S_i \\ S_j"] + list(map(lambda x: str(x), self.states)))

        # format tables
        ti = terminaltables.DoubleTable(initial, "Starting state counts")
        te = terminaltables.DoubleTable(emissions, "Emission counts")
        tt = terminaltables.DoubleTable(transitions, "Transition counts")
        ti.padding_left = 1
        ti.padding_right = 1
        te.padding_left = 1
        te.padding_right = 1
        tt.padding_left = 1
        tt.padding_right = 1
        ti.justify_columns[0] = "right"
        te.justify_columns[0] = "right"
        tt.justify_columns[0] = "right"

        # print tables
        print("\n")
        print(ti.table)
        print("\n")
        print(te.table)
        print("\n")
        print(tt.table)
        print("\n")

        #
        return None

    def print_probabilities(self):
        """
        Prints a copy of the current probabilities.
        Used for convenient checking in a command line environment.
        For dictionaries containing the raw values, use the `p_*` attributes.
        :return:
        """
        # create copies to avoid editing
        p_initial = copy.deepcopy(self.p_initial)
        p_emission = copy.deepcopy(self.p_emission)
        p_transition = copy.deepcopy(self.p_transition)

        # convert to nested lists for clean printing
        p_initial = [[str(s)] + [str(round(p_initial[s], 3))] for s in self.states]
        p_emission = [[str(s)] + [str(round(p_emission[s][e], 3)) for e in self.emissions]
            for s in self.states]
        p_transition = [[str(s1)] + [str(round(p_transition[s1][s2], 3)) for s2 in self.states]
            for s1 in self.states]

        p_initial.insert(0, ["S_i", "Y_0"])
        p_emission.insert(0, ["S_i \\ E_j"] + [str(e) for e in self.emissions])
        p_transition.insert(0, ["S_i \\ E_j"] + [str(s) for s in self.states])

        # format tables
        ti = terminaltables.DoubleTable(p_initial, "Starting state probabilities")
        te = terminaltables.DoubleTable(p_emission, "Emission probabilities")
        tt = terminaltables.DoubleTable(p_transition, "Transition probabilities")
        te.padding_left = 1
        te.padding_right = 1
        tt.padding_left = 1
        tt.padding_right = 1
        te.justify_columns[0] = "right"
        tt.justify_columns[0] = "right"

        # print tables
        print("\n")
        print(ti.table)
        print("\n")
        print(te.table)
        print("\n")
        print(tt.table)
        print("\n")

        #
        return None

    def calculate_chain_loglikelihood(self):
        
        """
        Calculate the negative log likelihood of the chain of emissions, given its current
        latent states. This is calculated based on the observed emission sequences only,
        and not on the probabilities of the hyperparameters.
        :return:
        """
        ### RETURN TO THIS ### not sure why we're negating
        return sum(chain.neglogp_chain(self.p_initial, self.p_emission, self.p_transition)
            for chain in self.chains)

    def calculate_loglikelihood(self):
        
        """
        Negative log-likelihood of the entire HDPHMM object. 
        Adds likelihoods of
         + the transition parameters
         + the emission parameters 
         + the chains themselves.
        Does not include the probabilities of the hyperparameter priors.
        :return: non-negative float
        """

        return (self.calculate_beta_mixture_variables_loglikelihood()
            + self.calculate_beta_emission_loglikelihood()
            + self.calculate_p_initial_loglikelihood()
            + self.calculate_p_transition_loglikelihood()
            + self.calculate_p_emission_loglikelihood()
            + self.calculate_chain_loglikelihood())

    def resample_chains(self, ncores=1):
        
        """
        Resample the latent states in all chains. This uses Beam sampling to improve the
        resampling time.
        :param ncores: int, number of threads to use in multithreading.
        :return: None
        """

        # extract probabilities
        p_initial, p_emission, p_transition = (self.p_initial,
        										self.p_emission,
        										self.p_transition)

        # create temporary function for mapping
        # Note: None state added here as possibility when resampling latent states
        resample_partial = functools.partial(Chain.resample_latent_sequence,
        									states=list(self.states) + [None],
            								p_initial=copy.deepcopy(p_initial),
            								p_emission=copy.deepcopy(p_emission),
            								p_transition=copy.deepcopy(p_transition))

        # parallel process beam resampling of latent states 
        ### RETURN TO THIS: each chain gets its separate process?###
        pool = multiprocessing.Pool(processes=ncores)
        new_latent_sequences = pool.map(resample_partial,((chain.emission_sequence, chain.latent_sequence) for chain in self.chains))
        pool.close()

        # assign returned latent sequences back to chains
        for i in range(self.c):
            
            self.chains[i].latent_sequence = new_latent_sequences[i]

        # call generator only on the Nones in the resampled latent sequences in order to assign new states to Nones
        # TODO: parameter check if we should be using alpha or gamma as parameter ### RETURN TO THIS: think this should be alpha
        state_generator = dirichlet_process_generator(self.hyperparameters["alpha"], output_generator=self.state_generator())
        for chain in self.chains:
            
            chain.latent_sequence = [s if s is not None else next(state_generator) for s in chain.latent_sequence]

        # update counts of 
        # + each emission per state
        # + transitions from each state to each other state, and 
        # + number of times each state is the initial
        self.update_counts()

    def maximise_hyperparameters(self):

        """
        Choose the MAP (maximum a posteriori) value for the hyperparameters.
        Not yet implemented.
        :return: None
        """

        raise NotImplementedError( "This has not yet been written!"
            + " Ping the author if you want it to happen.")
        
        pass

    def resample_hyperparameters(self):
        
        """
        Resample hyperparameters using a Metropolis Hastings algorithm. Uses a
        straightforward resampling approach, which (for each hyperparameter) samples a
        proposed value according to the prior distribution, and accepts the proposed
        value with probability scaled by the likelihood of the data given model under
        the current and proposed models.
        :return: None
        """

        # iterate and accept each in order
        for param_name in self.priors.keys():
            
            # skipa kappa if not stick HDPHMM
            if param_name == "kappa" and not self.sticky:
                continue

            # assign current param to object 
            param_current = self.hyperparameters[param_name]
            # get current negative log likelihood
            posterior_curr = self.calculate_loglikelihood()

            # draw from prior and assign to class variable
            self.hyperparameters[param_name] = self.priors[param_name]() ###RETURN TO THIS ### do we need to sample all at once?
            # calculate negative log likelihood given new params
            posterior_new = self.calculate_loglikelihood()

            # find Metropolis Hasting acceptance probability
            if posterior_new > posterior_curr:

            	proposed_accepted = True

            else:

            	p_accept = posterior_new / posterior_curr:

            	# draw from binom with prop p_accept to determine whether we accept posterior_new
            	alpha_accepted = bool(np.random.binomial(n=1, p=p_accept))

            # if we do not accept, revert to the previous value
            if not alpha_accepted:

                self.hyperparameters[param_name] = param_current

    def mcmc(self, n_iter=1000, burn_in=500, save_every=10, ncores=1, verbose=True):
        
        """
        Use Markov Chain Monte Carlo to estimate the starting, transition, and emission
        parameters of the HDPHMM, as well as the number of latent states.
        :param n_iter: int, number of iterations to complete.
        :param burn_in: int, number of iterations to complete before savings results.
        :param save_every: int, only iterations which are a multiple of `save_every`
        will have their results appended to the results.
        :param ncores: int, number of cores to use in multithreaded latent state
        resampling.
        :param verbose: bool, flag to indicate whether iteration-level statistics should
        be printed.
        :return: A dict containing results from every saved iteration. Includes:
          + the number of states of the HDPHMM
          + the negative log likelihood of the entire model
          + the negative log likelihood of the chains only
          + the hyperparameters of the HDPHMM
          + the emission beta values
          + the transition beta values
          + all probability dictionary objects
        """

        # store hyperparameters in a single dict
        results = {"state_count": list(),
        		"loglikelihood": list(),
        		"chain_loglikelihood": list(),
        		"hyperparameters": list(),
        		"beta_emission": list(),
        		"beta_mixture_variables": list(),
        		"parameters": list()}

        # cycle through iterations
        for i in tqdm.tqdm(range(n_iter)):

            """
            work down hierarchy when resampling
            """

            # resample latent states z_t via beam sampling
        	self.resample_chains(ncores=ncores)
            # remove unused latent states from previous iteration 
        	self.remove_unused_states()
            # resample beta_mixture_variables
            self.resample_beta_mixture_variables(ncores=ncores)
            # resample transition variables
            self.resample_p_initial()
            self.resample_p_transition()
            # resample emissions parameters
            self.resample_beta_emission()
            self.resample_p_emission()
            # resample priors alpha, gamma, kappa, emission priors
            self.resample_hyperparameters() # MH or Gibbs?

            # update computation-heavy statistics
            likelihood_curr = self.calculate_loglikelihood()

            # print iteration summary if required
            if verbose:
                if i == burn_in:
                    tqdm.tqdm.write("Burn-in iteration complete")
                
                # determines states removed and states added
                states_taken = states_prev - self.states
                states_added = self.states - states_prev
                

                msg = ["Iter: {}".format(i),
                	"Likelihood: {0:.1f}".format(likelihood_curr),
                    "states: {}".format(len(self.states))]
                
                if len(states_added) > 0:
                    msg.append("states added: {}".format(states_added))
                
                if len(states_taken) > 0:
                    msg.append("states removed: {}".format(states_taken))
                
                tqdm.tqdm.write(", ".join(msg))

            # store results
            if i >= burn_in and i % save_every == 0:
                
                # get hyperparameters as nested lists
                p_initial = copy.deepcopy(self.p_initial)
                p_emission = copy.deepcopy(self.p_emission)
                p_transition = copy.deepcopy(self.p_transition)

                # save new data
                results["state_count"].append(self.k)
                results["loglikelihood"].append(likelihood_curr)
                results["chain_loglikelihood"].append(self.calculate_chain_loglikelihood())

                results["hyperparameters"].append(copy.deepcopy(self.hyperparameters))
                results["beta_emission"].append(self.beta_emission)
                results["beta_mixture_variables"].append(self.beta_mixture_variables)
                results["parameters"].append({"p_initial": p_initial,
                							"p_emission": p_emission,
                        					"p_transition": p_transition})

        # return saved observations
        return results
