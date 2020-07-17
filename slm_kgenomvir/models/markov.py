from .bayes import _BaseBayes, check_alpha
import numpy as np


__author__ = "amine"

# #########################
#
# MARKOV CHAIN CLASSIFIERS
#
# #########################

class BaseMarkovChain(_BaseBayes):
    """
    """

    def _initial_fit(self, X, y):
        """
        """

        # fit the priors
        self._class_prior_fit(y)
 
        X_next = X[:, 0:self.v_size_]
        X_prev = X[:, self.v_size_:]

        # Compute y per target value
        self.count_per_class_next_ = np.zeros((self.n_classes_, self.v_size_)) 
        self.count_per_class_prev_ = np.zeros((self.n_classes_, X_prev.shape[1])) 

        for ind in range(self.n_classes_):
            X_next_class = X_next[y == self.classes_[ind]]
            X_prev_class = X_prev[y == self.classes_[ind]]
            # sum word by word
            self.count_per_class_next_[ind, :] = np.sum(X_next_class, axis=0)
            self.count_per_class_prev_[ind, :] = np.sum(X_prev_class, axis=0)

        return self
 
    def _log_joint_prob_density(self, X):
        """
        Compute the unnormalized posterior log probability of sequence
 
        I.e. ``log P(C) + log P(sequence | C)`` for all rows x of X, as an array-like of
        shape [n_sequences, n_classes].

        Input is passed to _log_joint_prob_density as-is by predict,
        predict_proba and predict_log_proba. 
        """

        X_next = X[:, 0:self.v_size_]
        X_prev = X[:, self.v_size_:]

        log_dot_next = np.dot(X_next, self.next_log_prob_.T)
        log_dot_prev = np.dot(X_prev, self.prev_log_prob_.T)

        return log_dot_next - log_dot_prev + self.log_class_priors_


class MLE_MarkovChain(BaseMarkovChain):
    """
    """

    def __init__(self, priors=None):
        self.priors = priors

    def fit(self, X, y, **kwargs):
        self.v_size_ = kwargs['v']
        y = np.asarray(y)
        self._initial_fit(X, y)

        with np.errstate(divide='ignore', invalid='ignore'):
            self.next_log_prob_ = np.nan_to_num(np.log(self.count_per_class_next_))
            self.prev_log_prob_ = np.nan_to_num(np.log(self.count_per_class_prev_)) 

        return self


class Bayesian_MarkovChain(BaseMarkovChain):
    """
    """

    def __init__(self, priors=None, alpha=1, alpha_classes=None): 
        self.priors = priors
        self.alpha = alpha
        self.alpha_classes = alpha_classes 

    def fit(self, X, y, **kwargs):
        self.v_size_ = kwargs['v']
        y = np.asarray(y)
        self._initial_fit(X, y)

        # validate alpha
        self.alpha = check_alpha(self.alpha)

        # Validate if the classes are the same as those estimated for alpha
        if self.alpha_classes is not None:
            if not np.array_equal(self.alpha_classes, self.classes_):
                raise ValueError("Classes from estimating alpha are not the same in y")

        alpha_main = alpha_back = self.alpha

        if isinstance(self.alpha, tuple):
            alpha_main = self.alpha[0]
            alpha_back = self.alpha[1]

        self.next_log_prob_ = np.log(self.count_per_class_next_ + alpha_main) 
        self.prev_log_prob_ = np.log(self.count_per_class_prev_ + alpha_back)

        return self

