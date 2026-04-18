"""
Thompson Sampling Bandit: Beta-distribution baseline for comparison with LinUCB.
"""

import numpy as np
import math
import logging
import pickle

logger = logging.getLogger(__name__)


class ThompsonSampling:
    """
    Beta-Thompson Sampling bandit for multi-arm selection.

    Each arm maintains a Beta(alpha_a, beta_a) distribution where:
        alpha_a = 1 + number of successes (reward >= threshold)
        beta_a  = 1 + number of failures  (reward <  threshold)

    Arm selection: sample theta_a ~ Beta(alpha_a, beta_a) for each arm,
                   pick argmax(theta_a).

    Reward binarisation: continuous rewards [0,1] are thresholded at 0.5
    to map onto the Beta distribution's Bernoulli assumption.
    """

    def __init__(self, n_arms: int = 3, reward_threshold: float = 0.5):
        """
        Args:
            n_arms: number of arms (should match LinUCB — 3 for Fast/Deep/Graph)
            reward_threshold: continuous rewards above this count as "success"
        """
        self.n_arms = n_arms
        self.reward_threshold = reward_threshold

        # Beta distribution parameters — initialised to Beta(1,1) = Uniform
        self.alpha = np.ones(n_arms)   # successes + 1
        self.beta  = np.ones(n_arms)   # failures  + 1

        # Step counter (mirrors LinUCB interface)
        self.t = 0

        # Per-arm raw reward history (for get_arm_performance, mirrors LinUCB)
        self.arm_rewards = [[] for _ in range(n_arms)]
        self.arm_window  = 50

    def select_arm(self, context=None) -> int:
        """
        Sample from each arm's Beta distribution and pick the highest.

        Args:
            context: ignored — Thompson Sampling is context-free.
                     Accepted to keep the same interface as LinUCB so
                     it can be dropped in anywhere LinUCB is used.

        Returns:
            int: selected arm index
        """
        samples = np.random.beta(self.alpha, self.beta)
        return int(np.argmax(samples))

    def select_arm_with_probs(self, context=None):
        """
        Select arm and return approximate selection probabilities.

        Probabilities are estimated by the ratio alpha/(alpha+beta) —
        the posterior mean for each arm — normalised to sum to 1.
        This is an approximation; true TS probabilities require
        Monte Carlo integration, which is overkill here.

        Returns:
            (selected_arm, probabilities, samples)
            - selected_arm: int
            - probabilities: np.array (n_arms,) — posterior means, normalised
            - samples: np.array (n_arms,) — the Beta samples used for selection
        """
        samples = np.random.beta(self.alpha, self.beta)
        selected_arm = int(np.argmax(samples))

        # Posterior means as proxy for selection probabilities
        means = self.alpha / (self.alpha + self.beta)
        probabilities = means / means.sum()

        return selected_arm, probabilities, samples

    def get_action_probabilities(self, context=None) -> np.ndarray:
        """
        Return normalised posterior means as arm selection probabilities.
        Mirrors LinUCB interface for use in off-policy evaluation.
        """
        means = self.alpha / (self.alpha + self.beta)
        return means / means.sum()

    def update(self, arm: int, context, reward: float):
        """
        Update Beta distribution for the selected arm.

        Args:
            arm: which arm was selected
            context: ignored (context-free algorithm)
            reward: observed reward in [0, 1]
        """
        # Binarise reward
        if reward >= self.reward_threshold:
            self.alpha[arm] += 1   # success
        else:
            self.beta[arm] += 1    # failure

        self.t += 1

        # Track raw reward history (for reporting, mirrors LinUCB)
        self.arm_rewards[arm].append(reward)
        if len(self.arm_rewards[arm]) > self.arm_window:
            self.arm_rewards[arm] = self.arm_rewards[arm][-self.arm_window:]

    def get_arm_performance(self) -> list:
        """Rolling average reward per arm. Mirrors LinUCB interface."""
        performances = []
        for arm in range(self.n_arms):
            if self.arm_rewards[arm]:
                performances.append(float(np.mean(self.arm_rewards[arm])))
            else:
                performances.append(0.5)
        return performances

    def get_posterior_means(self) -> np.ndarray:
        """Return the posterior mean reward estimate for each arm."""
        return self.alpha / (self.alpha + self.beta)

    def save_weights(self, path: str):
        """Save bandit state to disk. Mirrors LinUCB interface."""
        state = {
            'n_arms':           self.n_arms,
            'reward_threshold': self.reward_threshold,
            'alpha':            self.alpha.tolist(),
            'beta':             self.beta.tolist(),
            't':                self.t,
            'arm_rewards':      self.arm_rewards,
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        logger.info(f"Thompson Sampling weights saved to {path} (step {self.t})")

    def load_weights(self, path: str) -> bool:
        """Load bandit state from disk. Mirrors LinUCB interface."""
        try:
            with open(path, 'rb') as f:
                state = pickle.load(f)
            self.alpha       = np.array(state['alpha'])
            self.beta        = np.array(state['beta'])
            self.t           = state['t']
            self.arm_rewards = state['arm_rewards']
            return True
        except FileNotFoundError:
            logger.warning(f"No weights found at {path}")
            return False

    def __repr__(self):
        means = self.get_posterior_means()
        return (f"ThompsonSampling(n_arms={self.n_arms}, t={self.t}, "
                f"posterior_means={np.round(means, 3)})")