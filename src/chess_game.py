import chess
import numpy as np

class ChessGame:
    def __init__(self):
        self.action_size = 4672  
        
        self.build_move_index()

    def get_initial_state(self):
        return chess.Board()

    def get_next_state(self, state, action_uci, player):
        next_state = state.copy()
        move = chess.Move.from_uci(action_uci)
        next_state.push(move)
        return next_state

    def get_valid_moves(self, state):
        # Returns a binary mask of legal moves across all possible UCI moves
        valid_moves = np.zeros(self.action_size, dtype=np.uint8)
        legal_moves = list(state.legal_moves)
        for move in legal_moves:
            idx = self.move_to_index(move)
            valid_moves[idx] = 1
        return valid_moves

    def get_value_and_terminated(self, state, last_action=None):
        if state.is_game_over():
            result = state.result()
            if result == "1-0":
                return 1, True
            elif result == "0-1":
                return -1, True
            else:
                return 0, True
        return 0, False

    def get_opponent_value(self, value):
        return -value
    
    def change_perspective(self, state, player):
        return state

    def move_to_index(self, move):
        # Maps a move to an index in fixed-size action space
        # This is a placeholder. You need a consistent move indexing scheme (e.g., AlphaZero-style)
        uci = move.uci()
        return self.uci_to_index.get(uci, 0)

    def index_to_move(self, index):
        # Converts an index back to a move
        uci = self.index_to_uci.get(index, "0000")
        return chess.Move.from_uci(uci)

    def build_move_index(self):
        # Build mappings from UCI strings to indices (for fixed action space)
        self.uci_to_index = {}
        self.index_to_uci = {}
        index = 0
        for from_sq in chess.SQUARES:
            for to_sq in chess.SQUARES:
                for promo in [None, chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                    try:
                        move = chess.Move(from_sq, to_sq, promotion=promo)
                        if move.uci() not in self.uci_to_index:
                            self.uci_to_index[move.uci()] = index
                            self.index_to_uci[index] = move.uci()
                            index += 1
                    except:
                        pass
        self.action_size = index
