import os 
import h5py
import chess
import numpy as np
import uuid

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
        sim_count = max(int(max_simulations * 0.2), min(sim_count, max_simulations))
        return max(1, min(sim_count, max_simulations))

    def play_game(self, initial_board, max_simulations=100):
        game_data = []
        board = initial_board.copy() or chess.Board()

        while not board.is_game_over():
            simulations = self._get_simulations_num(board, max_simulations)
            move, policy, value = self._engine.best_move(board, simulations)
            
            state = self._engine.board_to_tensor(board)
            game_data.append((state, policy, value))

            board.push(move)
        
        return game_data
        
    def generate_games(self, num_games=10, max_simulations=100):
        games_data = []

        for _ in range(num_games):
            game_data = self.play_game(chess.Board(), max_simulations)  
            games_data.append(game_data)  

        return games_data

    def save_games(self, games_data, save_folder):
        if not save_folder:
            raise ValueError("path not provided.")

        save_folder = os.path.join(os.getcwd(), save_folder)
        os.makedirs(save_folder, exist_ok=True)

        file_uuid = str(uuid.uuid4())
        save_path = os.path.join(save_folder, f"{file_uuid}.h5")

        with h5py.File(save_path, "w") as h5_file:
            for game_idx, game_data in enumerate(games_data):
                game_group = h5_file.create_group(f"game_{game_idx}")

                states = np.array([step[0] for step in game_data], dtype=np.float32)
                policies = np.array([step[1] for step in game_data], dtype=np.float32)
                values = np.array([step[2] for step in game_data], dtype=np.float32)

                game_group.create_dataset("states", data=states, compression="gzip")
                game_group.create_dataset("policies", data=policies, compression="gzip")
                game_group.create_dataset("values", data=values, compression="gzip")
        
        print(f"Games saved successfully in: {save_path}")
        return save_path