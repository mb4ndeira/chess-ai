import math
import numpy as np

def calculate_ucb(C, visit_count,  action_value_sum, action_visit_count, action_prior):
    if action_visit_count == 0:
        q_value = 0
    else:
        q_value = 1 - ((action_value_sum / action_visit_count) + 1) / 2

    return q_value + C * (math.sqrt(visit_count) / (action_visit_count + 1)) * action_prior

class Node:
    def __init__(self, game, state, C, parent=None, action_taken=None, prior=0):
        self.game = game
        self.C = C
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.prior = prior
        
        self.best_policies = {}
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0

        self.perspective = game.get_perspective(state)
        
    def select(self):
        best_action = None
        best_ucb = -np.inf

        for action in self.best_policies:
            action_value_sum = self.children[action].value_sum if action in self.children else 0
            action_visit_count = self.children[action].visit_count if action in self.children else 0

            ucb = calculate_ucb(self.C, self.visit_count, action_value_sum, action_visit_count, self.best_policies[action])

            if ucb > best_ucb:
                best_action = action
                best_ucb = ucb

        if best_action not in self.children:
            next_state = self.game.get_next_state(self.state, self.game.index_to_move(best_action))
            self.children[best_action] = Node(self.game, next_state, self.C, self, best_action, self.best_policies[best_action])

        return self.children[best_action]

    def expand(self, policy, k_top=10):
        valid_moves = self.game.get_valid_moves(self.state)
        masked_policy = policy * valid_moves

        top_policies = sorted(
            [(i, masked_policy[i]) for i in range(len(masked_policy)) if masked_policy[i] > 0],
            key=lambda x: x[1],
            reverse=True
        )[:k_top]

        self.best_policies = {index: prob for index, prob in top_policies}
        
    def backpropagate(self, value):
        self.value_sum += value
        self.visit_count += 1

        if self.parent is not None:
            if self.parent.perspective != self.perspective:
                value = -value
            self.parent.backpropagate(value)


class MCTS: 
    def __init__(self, engine, game):
        self._game = game
        self._evaluate = engine.evaluate

    def search(self, state, simulations=100, C=2):
        root = Node(self._game, state, C)
        
        for _ in range(simulations):
            node = root

            while node.best_policies:
                node = node.select()

            is_terminal, value, policy = self._evaluate(node.state, node.perspective)

            if not is_terminal:
                node.expand(policy)

            node.backpropagate(value)

        action_probs = np.zeros(self._game.action_size)
        for child in root.children.values():
            action_probs[child.action_taken] = child.visit_count
        action_probs /= np.sum(action_probs)
        return action_probs
        