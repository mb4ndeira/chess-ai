from model.gaming_model import GamingRLModel
from chess_trainer import ChessTrainer
from engine import ChessEngine

def generate(model_path=None, games_data_path=None, num_games=10, max_simulations=100):
    model_path = model_path if model_path else "../gaming_model.keras"
    games_data_path = games_data_path if games_data_path else "../games_data"

    model = GamingRLModel(model_path)
    engine = ChessEngine(model)
    trainer = ChessTrainer(engine)

    results = trainer.generate_games("../games_data", num_games, max_simulations)

    return results