import math
import numpy as np
import time

class Node:
    def __init__(self, game, state, C, parent=None, action_taken=None, prior=0):
        self.game = game
        self.C = C
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.prior = prior
        
        self.children = []
        self.visit_count = 0
        self.value_sum = 0

        self.perspective = game.get_perspective(state)
        
    def select(self):
        best_child = None
        best_ucb = -np.inf

        for child in self.children:
            ucb = self.get_ucb(child)
            if ucb > best_ucb:
                best_child = child
                best_ucb = ucb

        return best_child

    def get_ucb(self, child):
        if child.visit_count == 0:
            q_value = 0
        else:
            q_value = 1 - ((child.value_sum / child.visit_count) + 1) / 2
        return q_value + self.C * (math.sqrt(self.visit_count) / (child.visit_count + 1)) * child.prior
    
    def expand(self, policy, k_top=10):
        valid_moves = self.game.get_valid_moves(self.state)
    
        masked_policy = policy * valid_moves

        top_indices = np.argpartition(-masked_policy, k_top)[:k_top]
        top_indices = [i for i in top_indices if masked_policy[i] > 0][:10]
            
        for action in top_indices:
            prob = policy[action]

            move = self.game.index_to_move(action)
            child_state = self.game.get_next_state(self.state, move)
            
            child = Node(self.game, child_state, self.C, self, action, prob)
            self.children.append(child)
        
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
        
        selection_time = 0.0
        evaluation_time = 0.0
        expand_time = 0.0
        backprop_time = 0.0

        for _ in range(simulations):
            sim_start = time.time()
            node = root

            while node.children:
                node = node.select()
            selection_time += time.time() - sim_start

            eval_start = time.time()
            is_terminal, value, policy = self._evaluate(node.state, node.perspective)
            evaluation_time += time.time() - eval_start

            if not is_terminal:
                expand_start = time.time()
                node.expand(policy)
                expand_time += time.time() - expand_start

            backprop_start = time.time()
            node.backpropagate(value)
            backprop_time += time.time() - backprop_start

        print(f"Total time over {simulations} sims:")
        print(f"  Selection:     {selection_time:.4f}s")
        print(f"  Evaluation:    {evaluation_time:.4f}s")
        print(f"  Expansion:     {expand_time:.4f}s")
        print(f"  Backpropagation: {backprop_time:.4f}s")

        action_probs = np.zeros(self._game.action_size)
        for child in root.children:
            action_probs[child.action_taken] = child.visit_count
        action_probs /= np.sum(action_probs)
        return action_probs
        