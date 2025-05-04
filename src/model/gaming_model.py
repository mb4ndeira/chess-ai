import os
import numpy as np
from tensorflow.config import list_physical_devices
from tensorflow.keras import  mixed_precision, models

from model.res_net import build_resnet

class GamingRLModel():
    def __init__(self, model_path=None):
        default_model_path = "gaming_model.keras"

        if model_path is not None:
            if os.path.exists(model_path):
                self._load_model(model_path)
            else:
                print("model path not found. defaulting to model_path: ", default_model_path)
                self._model_path = default_model_path
                self._model = self._build_model()

        if model_path is None:
            print("model path not provided. defaulting to model_path: ", default_model_path)
            self._model_path = default_model_path
            self._model = self._build_model()

        if list_physical_devices('GPU'):
            mixed_precision.set_global_policy('mixed_float16')

    def _load_model(self, model_path):
        try:
            self._model = models.load_model(model_path)
        except Exception as e:
            print(f"error loading model: {e}")
            self._model = None

    def _save_model(self, save_path):
        try:
            self._model.save(save_path)
            print(f"model saved at {save_path}")
        except Exception as e:
            print(f"error saving model: {e}")
            self.model = None

    def _build_model(self):
        input_shape = (8, 8, 16)
        action_size = 4672  
        num_resBlocks = 3  
        num_hidden = 64  

        model = build_resnet(input_shape, action_size, num_resBlocks, num_hidden)
        return model

    def train(self, train_data, epochs=10, batch_size=64, validation_data=None):
        """
        Train the model using the provided data.

        train_data: Tuple containing the training data (inputs, (policy_target, value_target)).
        """
        
        X_train, y_train = train_data
        policy_target, value_target = y_train
        
        self._model.compile(optimizer='adam', loss=['categorical_crossentropy', 'mean_squared_error'])

        history = self._model.fit(
            X_train, [policy_target, value_target],
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data
        )
        
        return history

    def predict(self, states):
        if len(states.shape) == 3:  
            states = np.expand_dims(states, axis=0)  

        policy, value = self._model.predict(states)
        return policy, value
