import chess
import numpy as np

class ChessTrainer:
    def __init__(self, engine):  
        self._engine = engine

    def _get_simulations_num(self, board, max_simulations):
        board_copy = board.copy()
        board_copy

        num_legal_moves = len(list(board_copy.legal_moves))
        num_moves_played = board_copy.fullmove_number * 2 - (0 if board.turn else 1)

        progress_ratio = min(num_moves_played / 80, 1.0) 
        branching_ratio = min(num_legal_moves / 40, 1.0)   

        adjustment_factor = (1 - branching_ratio + progress_ratio) / 2

        sim_count = int(max_simulations * adjustment_factor)
        return max(1, min(sim_count, max_simulations))

    def play_game(self, board, max_simulations=100):
        game_data = []

        if board is None:
            board = chess.Board()

        while not board.is_game_over():
            simulations = self._get_simulations_num(board, max_simulations)
            print(simulations)
            move, policy, value = self._engine.best_move(board, simulations)
            
            state = self._engine.board_to_tensor(board)
            game_data.append((state, policy, value))

            board.push(move)
        
        return game_data
        
    def generate_games(self, num_games=10):
        games_data = []

        for _ in range(num_games):
            game_data = self.play_game(chess.Board())  
            games_data.extend(game_data)  

        return games_data

    #  def encode_piece(self, piece):
    #     """
    #     Encode the piece type and color into a vector.
    #     This is a simple encoding — you can expand it as needed for your model.
    #     """
    #     piece_map = {
    #         chess.PAWN: 1,
    #         chess.KNIGHT: 2,
    #         chess.BISHOP: 3,
    #         chess.ROOK: 4,
    #         chess.QUEEN: 5,
    #         chess.KING: 6
    #     }
        
    #     # Encoding: 0 for empty, positive for white pieces, negative for black pieces
    #     piece_value = np.zeros(3)  # 3 channels: piece type, color, etc.
    #     if piece:
    #         piece_value[0] = piece_map.get(piece.piece_type, 0)
    #         piece_value[1] = 1 if piece.color == chess.WHITE else -1  # Encoding color as +1 for white, -1 for black
    #     return piece_value

    # def get_action_index_from_move(self, state, move):
    #     """
    #     Convert the move into an action index for the policy head.
    #     You should map the move into an index in your action space.
    #     This can be tricky depending on how you're encoding moves.
    #     """
    #     # This function should return the index corresponding to the move in the action space
    #     # For now, let's assume it's a dummy function:
    #     return np.random.randint(4672)  # Random index as a placeholder