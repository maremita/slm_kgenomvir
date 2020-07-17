from abc import ABC, abstractmethod

import numpy as np
from scipy.special import logsumexp, gammaln

from sklearn.base import BaseEstimator, ClassifierMixin


__author__ = "amine"

# #########################
#
# BASE CLASSIFIER
#
# #########################


# TODO
def check_alpha(alpha):

    if isinstance(alpha, list):
        alpha = np.asarray(alpha)

    return alpha


class _BaseBayes(ABC, BaseEstimator, ClassifierMixin):
    """
    Do not instantiate this class
    """

    def _class_prior_fit(self, y):
        """
        """
        
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        # Compute class priors
        # Calcul des probabilites a priori pour chaque classe

        self.class_counts_ = np.zeros(self.n_classes_)

        for ind in range(self.n_classes_):
            self.class_counts_[ind] = len(y[y==self.classes_[ind]])
 
        if self.priors == "uniform":
            self.class_priors_ = np.full(self.n_classes_, 1/self.n_classes_)

        elif self.priors == "ones":
            self.class_priors_ = np.full(self.n_classes_, 1)

        elif self.priors is not None:
            self.class_priors_ = self.priors

        else:
            self.class_priors_ = self.class_counts_ / self.class_counts_.sum()

        # log class priors
        self.log_class_priors_ = np.log(self.class_priors_)

        return self
 
    @abstractmethod
    def _log_joint_prob_density(self, X):
        """
        Compute the unnormalized posterior log probability of sequence
 
        I.e. ``log P(C) + log P(sequence | C)`` for all rows x of X, as an array-like of
        shape [n_sequences, n_classes].

        Input is passed to _log_joint_prob_density as-is by predict,
        predict_proba and predict_log_proba. 
        """

    def predict(self, X):
        """
        Perform classification on an array of test vectors X.

        Parameters
        ----------
        X : 

        Returns
        -------
        C : array, shape = (n_sequences)
            Predicted target values for X
        """

        ljb = self._log_joint_prob_density(X)
        # ljb has a shape of (n_sequences, n_classes)

        return self.classes_[np.argmax(ljb, axis=1)]

    def predict_log_proba(self, X):
        """
        Return log-probability estimates for the test vector X.
        
        Parameters
        ----------
        X :
 
        Returns
        -------
        C : array-like, shape = (n_sequences, n_classes)
            Returns the log-probability of the sequences for each class in
            the model. The columns correspond to the classes in sorted
            order, as they appear in the attribute `classes_`.
        """

        ljb = self._log_joint_prob_density(X)

        # normalize by P(x) = P(f_1, ..., f_n) la formule est fausse
        # We use marginalization to compute P(x)
        # P(x) = sum(P(x, c_i))

        # ljb contains log joint prob densitys for each class
        # We put this values in exp and sum them
        # finally, we compute the log of the sum
        # logsumexp : Compute the log of the sum of exponentials of 
        #             input elements.
        #
        # we substract log_prob_x because we calculate with logs, if we work
        # with densities or probabilities we divide by P(x)

        log_prob_x = logsumexp(ljb, axis=1)

        return ljb - np.atleast_2d(log_prob_x).T

    def predict_proba(self, X):
        """
        Return probability estimates for the test vector X.
 
        Parameters
        ----------
        X :

        Returns
        -------
        C : array-like, shape = (n_sequences, n_classes)
            Returns the probability of the sequences for each class in
            the model. The columns correspond to the classes in sorted
            order, as they appear in the attribute `classes_`.
        """

        return np.exp(self.predict_log_proba(X))

# #########################
#
# MULTINOMIAL BAYES MODELS
#
# #########################

class BaseMultinomialBayes(_BaseBayes):
    """
    """

    def _initial_fit(self, X, y):
        """
        """

        # fit the priors
        self._class_prior_fit(y)

        self.v_size_ = X.shape[1]

        # compute y per target value
        self.count_per_class_ = np.zeros((self.n_classes_, self.v_size_)) 
        self.kmer_log_probs_ = np.zeros(self.count_per_class_.shape)

        for ind in range(self.n_classes_):
            X_class = X[y == self.classes_[ind]]
            # sum word by word
            self.count_per_class_[ind, :] = np.sum(X_class, axis=0)

        # compute the sum of ys
        self.total_counts_per_class_ = self.count_per_class_.sum(axis=1)

        return self

    def _log_joint_prob_density(self, X):
        """
        Compute the unnormalized posterior log probability of sequence
 
        I.e. ``log P(C) + log P(sequence | C)`` for all rows x of X, as an array-like of
        shape [n_sequences, n_classes].

        Input is passed to _log_joint_prob_density as-is by predict,
        predict_proba and predict_log_proba. 
        """

        #log_joint_prob_density = []
        #for i in range(n_classes):
        #    # compute the log conditional prob density distribution for class i
        #    log_likelihood_dens = np.dot(X, self.kmer_log_probs_[i])
        #    # compute the log joint prob density distribution for class i
        #    log_joint_prob_density.append(self.log_class_priors_[i] + log_likelihood_dens)
        #log_joint_prob_density = np.array(log_joint_prob_density).T
 
        log_cte_norm = gammaln(X.sum(axis=1) + 1) - gammaln(X+1).sum(axis=1)
        log_dot_prob = np.dot(X, self.kmer_log_probs_.T)

        return log_dot_prob + log_cte_norm.reshape(1, -1).T + self.log_class_priors_
        # return log_dot_prob + self.log_class_priors_


class MLE_MultinomialBayes(BaseMultinomialBayes):
    """
    """
 
    def __init__(self, priors=None):
        self.priors = priors

    def fit(self, X, y):
        y = np.asarray(y)
        self._initial_fit(X, y)

        # Method 1
        #for ind in range(n_classes):
        #    self.kmer_log_probs_[ind] = np.log(self.count_per_class_[ind]) - np.log(self.total_counts_per_class_[ind])
        #self.kmer_log_probs_ = np.nan_to_num(self.log_kmer_probs_)

        # Method 2
        #self.kmer_log_probs_ = np.nan_to_num(np.log(self.count_per_class_.T) - np.log(self.total_counts_per_class_)).T
        
        # Method 3
        with np.errstate(divide='ignore', invalid='ignore'):
            self.kmer_log_probs_ = np.nan_to_num(np.log(self.count_per_class_) - np.log(self.total_counts_per_class_.reshape(-1, 1))) 
            #self.kmer_log_probs_ = np.log(self.count_per_class_) - np.log(self.total_counts_per_class_.reshape(-1, 1)) 

        return self


class Bayesian_MultinomialBayes(BaseMultinomialBayes):
    """
    """

    def __init__(self, priors=None, alpha=None, alpha_classes=None): 
        self.priors = priors
        self.alpha = alpha
        self.alpha_classes = alpha_classes 

    def fit(self, X, y):
        y = np.asarray(y)
        self._initial_fit(X, y)

        # validate alpha
        self.alpha = check_alpha(self.alpha)
 
        # Validate if the classes are the same as those estimated for alpha
        if self.alpha_classes is not None:
            if not np.array_equal(self.alpha_classes, self.classes_):
                raise ValueError("Classes from estimating alpha are not the same in y")

        # Beta
        self.beta_ = self.count_per_class_ + self.alpha
        #beta_sum = self.total_counts_per_class_ + (self.alpha * self.v_size_)
        beta_sum = self.beta_.sum(axis=1) 
        
        #print(self.count_per_class_.sum(axis=1))
        #print(self.beta_.sum(axis=1))

        self.kmer_log_probs_ = np.log(self.beta_) - np.log(beta_sum.reshape(-1, 1))

        return self

