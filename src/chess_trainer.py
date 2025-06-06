import os 
import h5py
import chess
import numpy as np
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed

def play_game_worker(func, max_simulations):
    return func(chess.Board(), max_simulations)

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
        board = initial_board.copy() if initial_board else chess.Board()

        while not board.is_game_over():
            simulations = self._get_simulations_num(board, max_simulations)
            move, policy, value = self._engine.best_move(board, simulations, 1.41)
            
            state = self._engine.board_to_tensor(board)
            game_data.append((state, policy, value))

            print(move)
            board.push(move)
        
        return game_data
        
    def generate_games(self, save_folder,  num_games=10, max_simulations=100):
        os.makedirs(save_folder, exist_ok=True)
        buffer = []

        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = [executor.submit(play_game_worker, self.play_game, max_simulations) for _ in range(num_games)]

            for future in as_completed(futures):
                game_memory = future.result()
                buffer.append(game_memory)

                if len(buffer) >= 10:
                    self.save_games(buffer, save_folder)
                    buffer = []

        if buffer:
            self.save_games(buffer, save_folder)

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
    
    def load_games(self, load_path):
        if not load_path or not os.path.exists(load_path):
            raise ValueError("Invalid path or folder does not exist.")
        
        loaded_games = []

        if os.path.isdir(load_path):
            h5_files = [os.path.join(load_path, file) for file in os.listdir(load_path) if file.endswith(".h5")]
            if not h5_files:
                raise ValueError("No .h5 files found in the specified directory.")
        else:
            h5_files = [load_path]

        for file in h5_files:
            with h5py.File(file, "r") as h5_file:
                for game_name in h5_file.keys():
                    game_group = h5_file[game_name]
                    
                    states = game_group["states"][:]
                    policies = game_group["policies"][:]
                    values = game_group["values"][:]

                    game_data = [(states[i], policies[i], values[i]) for i in range(len(states))]
                    loaded_games.append(game_data)
        
        print(f"Loaded {len(loaded_games)} games from {load_path}")
        return loaded_games

    def delete_games(delete_path, force=False):
        if not delete_path or not os.path.exists(delete_path):
            raise ValueError("Invalid path or folder does not exist.")
        
        if os.path.isdir(delete_path):
            h5_files = [os.path.join(delete_path, file) for file in os.listdir(delete_path) if file.endswith(".h5")]
        else:
            h5_files = [delete_path] if delete_path.endswith(".h5") else []

        if not h5_files:
            raise ValueError("No .h5 files found to delete.")

        if not force:
            print(f"Found {len(h5_files)} .h5 file(s) to delete:")
            for f in h5_files:
                print(" -", f)
            confirm = input("Are you sure you want to delete these files? [y/N]: ")
            if confirm.lower() != 'y':
                print("Deletion canceled.")
                return

        for file in h5_files:
            try:
                os.remove(file)
                print(f"Deleted: {file}")
            except Exception as e:
                print(f"Failed to delete {file}: {e}")

        print("Deletion completed.")