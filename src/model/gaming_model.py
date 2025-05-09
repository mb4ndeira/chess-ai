import os
import numpy as np
from tensorflow.config import list_physical_devices
from tensorflow.keras import  mixed_precision, models

from model.res_net import build_resnet

class GamingRLModel():
    def __init__(self, model_path=None):
        default_model_path = "gaming_model.keras"

        if model_path is not None:
            model_path = os.path.join(os.getcwd(), model_path)  
            if os.path.exists(model_path):
                self._load_model(model_path)
            else:
                print("Model path not found. Defaulting to model_path:", default_model_path)
                self._model_path = os.path.join(os.getcwd(), default_model_path)
                self._model = self._build_model()

        if model_path is None:
            print("Model path not provided. Defaulting to model_path:", default_model_path)
            self._model_path = os.path.join(os.getcwd(), default_model_path)
            self._model = self._build_model()

        if list_physical_devices('GPU'):
            mixed_precision.set_global_policy('mixed_float16')

    def _load_model(self, model_path):
        try:
            self._model = models.load_model(model_path)
            print(f"Model loaded from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self._model = None

    def _save_model(self, save_path):
        save_path = os.path.join(os.getcwd(), save_path) 
        try:
            self._model.save(save_path)
            print(f"Model saved at {save_path}")
        except Exception as e:
            print(f"Error saving model: {e}")
            self._model = None

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
        
        X_train = np.array([step[0] for step in train_data], dtype=np.float32)
        policy_target = np.array([step[1] for step in train_data], dtype=np.float32)
        value_target = np.array([step[2] for step in train_data], dtype=np.float32)
        
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

        if states.shape[0] == 1:
            policy = np.squeeze(policy, axis=0)
            value = np.squeeze(value, axis=0)

        return policy, value
