import os
from backblaze_gateway import BackblazeGateway

class CloudModelSaver:
    def __init__(self, save_path, gateway: BackblazeGateway):
        self.save_path = save_path
        self.gateway = gateway

    def save(self, model):
        temp_path = "/tmp/model_file.keras"

        try:
            model.save(temp_path)

            self.gateway.upload_file(temp_path, self.save_path)
            print(f"Model uploaded to the cloud.")

        except Exception as e:
            print(f"Error saving model to cloud: {e}")
            model = None

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

        return model
