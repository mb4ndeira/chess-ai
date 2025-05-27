import argparse
from scripts.train import train as train_script
from scripts.generate import generate as generate_script

def main():
    parser = argparse.ArgumentParser(description="Train or generate with the GamingRLModel.")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--model-path", type=str, default="gaming_model.keras", help="Path to the model file")
    train_parser.add_argument("--games-data-path", type=str, default="games_data", help="Path to training data")
    train_parser.add_argument("--delete-games", type=bool, default=False, help="Delete games after training")

    generate_parser = subparsers.add_parser("generate", help="Generate self-play games")
    generate_parser.add_argument("--model-path", type=str, default="gaming_model.keras", help="Path to the model file")
    generate_parser.add_argument("--games-data-path", type=str, default="games_data", help="Directory to save generated games")
    generate_parser.add_argument("--num-games", type=int, default=10, help="Number of games to generate")
    generate_parser.add_argument("--max-simulations", type=int, default=100, help="Maximum simulations per move")

    args = parser.parse_args()

    if args.mode == "train":
        print("Training mode selected.")
        train_script(model_path=args.model_path, games_data_path=args.games_data_path, delete_games=args.delete_games)
        print("Training completed.")
    elif args.mode == "generate":
        print("Generation mode selected.")
        generate_script(
            model_path=args.model_path,
            games_data_path=args.games_data_path,
            num_games=args.num_games,
            max_simulations=args.max_simulations
        )
        print("Generation completed.")

if __name__ == "__main__":
    main()
