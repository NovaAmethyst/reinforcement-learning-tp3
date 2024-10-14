from qlearning import QLearningAgent

import typing as t
import numpy as np
import random

Action = int
State = int
Info = t.TypedDict("Info", {"prob": float, "action_mask": np.ndarray})
QValues = t.DefaultDict[int, t.DefaultDict[Action, float]]

class QLearningAgentEpsReduce(QLearningAgent):
    def __init__(
        self,
        learning_rate: float,
        epsilon: float,
        gamma: float,
        legal_actions: t.List[Action],
    ):
        super().__init__(learning_rate, epsilon, gamma, legal_actions)

    def get_action(self, state):
        rand = random.uniform(0.0, 1.0)
        action = self.get_best_action(state)
        if self.get_qvalue(state, action) <= 1:
            random_choice_proba = self.epsilon
        else:
            random_choice_proba = self.epsilon / self.get_qvalue(state, action)
        if rand < random_choice_proba:
            action = random.choice(self.legal_actions)
        return action