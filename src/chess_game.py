import chess
import numpy as np

class ChessGame:
    def __init__(self):
        self.action_size = 4672  
        
        self._move_to_index_map, self._index_to_move_map = self._build_move_index()

    def _build_move_index(self):
        QUEEN_DIRS = [(-1, 0), (-1, 1), (0, 1), (1, 1),
                    (1, 0), (1, -1), (0, -1), (-1, -1)]

        KNIGHT_DIRS = [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
                    (1, -2), (1, 2), (2, -1), (2, 1)]

        index_to_move_map = {}
        move_to_index_map = {}

        for from_square in range(64):
            from_rank = chess.square_rank(from_square)
            from_file = chess.square_file(from_square)

            move_type = 0

            # Queen-like moves (8 directions × up to 7 steps)
            for dx, dy in QUEEN_DIRS:
                for step in range(1, 8):
                    to_file = from_file + dx * step
                    to_rank = from_rank + dy * step
                    if 0 <= to_file < 8 and 0 <= to_rank < 8:
                        to_square = chess.square(to_file, to_rank)
                        move = chess.Move(from_square, to_square)
                        index = from_square * 73 + move_type
                        index_to_move_map[index] = move
                        move_to_index_map[move] = index
                        move_type += 1
                    else:
                        break

            # Knight moves
            for dx, dy in KNIGHT_DIRS:
                to_file = from_file + dx
                to_rank = from_rank + dy
                if 0 <= to_file < 8 and 0 <= to_rank < 8:
                    to_square = chess.square(to_file, to_rank)
                    move = chess.Move(from_square, to_square)
                    index = from_square * 73 + move_type
                    index_to_move_map[index] = move
                    move_to_index_map[move] = index
                    move_type += 1

            # Promotions (only from rank 6 for white, rank 1 for black)
            # We'll add promotions to q, r, b, n for forward, capture-left, capture-right
            promotion_pieces = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
            directions = [0, -1, 1]  # forward, capture-left, capture-right

            for color in [chess.WHITE, chess.BLACK]:
                is_white = color == chess.WHITE
                promotion_rank = 6 if is_white else 1
                next_rank = 7 if is_white else 0
                if from_rank != promotion_rank:
                    continue

                for d_file in directions:
                    to_file = from_file + d_file
                    if not (0 <= to_file < 8):
                        continue
                    to_square = chess.square(to_file, next_rank)

                    for piece in promotion_pieces:
                        move = chess.Move(from_square, to_square, promotion=piece)
                        index = from_square * 73 + move_type
                        index_to_move_map[index] = move
                        move_to_index_map[move] = index
                        move_type += 1

        return move_to_index_map, index_to_move_map

    def get_initial_state(self):
        return chess.Board()

    def get_next_state(self, state, action_uci):
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
    
    def get_perspective(self, state):
        return "white" if state.turn else "black"

    def move_to_index(self, move):
        return self._move_to_index_map.get(move)

    def index_to_move(self, index):
        return self._index_to_move_map.get(index)



