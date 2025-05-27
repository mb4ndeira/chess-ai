import os
from datetime import datetime
from dotenv import load_dotenv

from model.gaming_model import GamingRLModel
from chess_trainer import ChessTrainer
from engine import ChessEngine

def train(model_path=None, games_data_path=None, delete_games=False, cloud_save=False):
    load_dotenv() 

    model_path = model_path if model_path else "../gaming_model.keras"
    games_data_path = games_data_path if games_data_path else "../games_data"

    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    filename = f"gaming_model_{timestamp}.keras"
    model_saver = None
    if cloud_save:
        from backblaze_gateway import BackblazeGateway  
        from model.cloud_model_saver import CloudModelSaver

        application_key_id = os.getenv("B2_APPLICATION_KEY_ID")
        application_key = os.getenv("B2_APPLICATION_KEY")
        bucket_name = os.getenv("B2_BUCKET_NAME")

        if not all([application_key_id, application_key, bucket_name]):
            raise ValueError("Missing Backblaze credentials in environment variables")

        gateway = BackblazeGateway(application_key_id, application_key, bucket_name)
        model_saver = CloudModelSaver(filename, gateway)
    else:    
        from model.local_model_saver import LocalModelSaver
        model_saver = LocalModelSaver(f"../{filename}")

    model = GamingRLModel(model_saver, model_path)
    engine = ChessEngine(model)
    trainer = ChessTrainer(engine)

    results = trainer.load_games(games_data_path)

    flat_results = [step for game in results for step in game]

    def callback():
        if delete_games:
            print("Deleting used games...")
            trainer.delete_games(games_data_path)

    model.train(flat_results, callback, 1)
