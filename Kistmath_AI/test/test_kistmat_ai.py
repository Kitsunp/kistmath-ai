import unittest
import numpy as np
import tensorflow as tf
import os
import tempfile
from models.kistmat_ai import Kistmat_AI
from config.settings import VOCAB_SIZE, MAX_LENGTH

class TestKistmatAI(unittest.TestCase):
    def setUp(self):
        self.model = Kistmat_AI(input_shape=(MAX_LENGTH,), output_shape=1)
        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    def test_model_structure(self):
        """Test the structure of the model"""
        self.assertIsInstance(self.model.model, tf.keras.Model)
        self.assertEqual(len(self.model.model.layers), 5)  # Adjust this number based on your actual model structure
        self.assertEqual(self.model.model.input_shape, (None, MAX_LENGTH))
        self.assertEqual(self.model.model.output_shape, (None, 1))

    def test_model_training(self):
        """Test model training and evaluation"""
        X_train = np.random.randint(0, VOCAB_SIZE, size=(100, MAX_LENGTH))
        y_train = np.random.rand(100, 1)

        history = self.model.fit(X_train, y_train, epochs=5, batch_size=10, verbose=0)

        self.assertIn('loss', history.history)
        self.assertIn('mae', history.history)
        
        result = self.model.predict(X_train[:1])
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, (1, 1))

        score = self.model.evaluate(X_train, y_train, verbose=0)
        self.assertGreaterEqual(score[1], 0.8)  # Assuming the second metric is the score

    def test_model_save_load(self):
        """Test saving and loading the model"""
        X_test = np.random.randint(0, VOCAB_SIZE, size=(10, MAX_LENGTH))
        
        original_predictions = self.model.predict(X_test)
        
        with tempfile.TemporaryDirectory() as tmpdirname:
            save_path = os.path.join(tmpdirname, 'test_model')
            self.model.save(save_path)
            
            loaded_model = Kistmat_AI.load(save_path)
            loaded_predictions = loaded_model.predict(X_test)
            
            np.testing.assert_array_almost_equal(original_predictions, loaded_predictions, decimal=6)

    def test_different_input_types(self):
        """Test model behavior with different types of input"""
        # Test with float input
        X_float = np.random.rand(10, MAX_LENGTH)
        result_float = self.model.predict(X_float)
        self.assertEqual(result_float.shape, (10, 1))

        # Test with integer input
        X_int = np.random.randint(0, VOCAB_SIZE, size=(10, MAX_LENGTH))
        result_int = self.model.predict(X_int)
        self.assertEqual(result_int.shape, (10, 1))

        # Test with boolean input
        X_bool = np.random.choice([True, False], size=(10, MAX_LENGTH))
        result_bool = self.model.predict(X_bool)
        self.assertEqual(result_bool.shape, (10, 1))

    def test_error_handling(self):
        """Test error handling and edge cases"""
        # Test with input of wrong shape
        X_wrong_shape = np.random.randint(0, VOCAB_SIZE, size=(10, MAX_LENGTH + 1))
        with self.assertRaises(ValueError):
            self.model.predict(X_wrong_shape)

        # Test with empty input
        X_empty = np.array([])
        with self.assertRaises(ValueError):
            self.model.predict(X_empty)

        # Test with NaN values
        X_nan = np.full((10, MAX_LENGTH), np.nan)
        result_nan = self.model.predict(X_nan)
        self.assertTrue(np.all(np.isnan(result_nan)))

    def test_prediction_consistency(self):
        """Test consistency of model predictions"""
        X_test = np.random.randint(0, VOCAB_SIZE, size=(10, MAX_LENGTH))
        
        predictions1 = self.model.predict(X_test)
        predictions2 = self.model.predict(X_test)
        
        np.testing.assert_array_almost_equal(predictions1, predictions2, decimal=6)

if __name__ == '__main__':
    unittest.main()