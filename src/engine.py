import chess
import numpy as np

from mcts import MCTS
from chess_game import ChessGame

class ChessEngine:
    def __init__(self, model):
        self._model = model
        self._game = ChessGame()

    def board_to_tensor(self, board):
        def encode_piece_type(piece):
            piece_type = piece.piece_type
            color_offset = 0 if piece.color == chess.WHITE else 6
            return color_offset + piece_type - 1
        
        state = np.zeros((8, 8, 16), dtype=np.float32)

        state[:, :, 12] = 1 if board.has_kingside_castling_rights(chess.WHITE) else 0
        state[:, :, 13] = 1 if board.has_queenside_castling_rights(chess.WHITE) else 0
        state[:, :, 14] = 1 if board.has_kingside_castling_rights(chess.BLACK) else 0
        state[:, :, 15] = 1 if board.has_queenside_castling_rights(chess.BLACK) else 0

        for row in range(8):
            for col in range(8):
                piece = board.piece_at(chess.square(col, 7 - row))
                if piece:
                    state[row, col, encode_piece_type(piece)] = 1

        return state

    def best_move(self, board, simulations=100, C=1.41): 
        action_probs = MCTS(self, self._game).search(board, simulations, C)

        best_action = np.argmax(action_probs)
        move = self._game.index_to_move(best_action)

        _, value = self._model.predict(self.board_to_tensor(board))
        # value = np.random.uniform(-1, 1)

        return move, action_probs, value
    
    def evaluate(self, state, perspective):
        def predict_policy_and_value(state):
            policy, value = self._model.predict(self.board_to_tensor(state))
            print(policy.shape)
            return policy, value
        
        # def randomize_policy_and_value(state):
        #     valid_moves = self._game.get_valid_moves(state)

        #     policy = np.random.rand(*valid_moves.shape)  
        #     if np.sum(valid_moves) == 0:
        #         policy = np.ones_like(valid_moves) / len(valid_moves)
        #     else:
        #         policy *= valid_moves
        #         if np.sum(policy) == 0:
        #             policy = valid_moves / np.sum(valid_moves)
        #         else:
        #             policy /= np.sum(policy)

        #     value = np.random.uniform(-1, 1)
        #     return policy, value
        
        if state.is_game_over():
            result = state.result()
            if result == "1-0":
                return True, 1, None  
            elif result == "0-1":
                return True, -1, None
            else:
                return True, 0, None 

        # policy, value = randomize_policy_and_value(state)
        policy, value = predict_policy_and_value(state)
        
        if perspective == 'black':
            value = -value

        return False, value, policy
