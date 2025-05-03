import chess
import numpy as np

from mcts import mcts
from chess_game import ChessGame

class ChessEngine:
    # def __init__(self):

    def best_move(self, board): 
        game = ChessGame()
        action_probs = mcts(board, game, simulations=100, C=2)

        best_action = np.argmax(action_probs)
        best_uci = game.index_to_move(best_action).uci()
        return chess.Move.from_uci(best_uci)
