from model.gaming_model import GamingRLModel
from chess_trainer import ChessTrainer
from engine import ChessEngine

def train(model_path=None, games_data_path=None):
    model_path = model_path if model_path else "../gaming_model.keras"
    games_data_path = games_data_path if games_data_path else "../games_data"

    model = GamingRLModel(model_path)
    engine = ChessEngine(model)
    trainer = ChessTrainer(engine)

    results = trainer.load_games(games_data_path)

    flat_results = [step for game in results for step in game]

    model.train(flat_results)