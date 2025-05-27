import os

class LocalModelSaver:
    def __init__(self, save_path):
        self.save_path = os.path.join(os.getcwd(), save_path) 

    def save(self, model):
        try:
            model.save(self.save_path)
            print(f"Model saved at {self.save_path}")
        except Exception as e:
            print(f"Error saving model: {e}")
            model = None
        
        return model
