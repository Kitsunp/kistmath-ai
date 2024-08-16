import unittest
import numpy as np
from Kistmath_AI.models.kistmat_ai import Kistmat_AI
from Kistmath_AI.config.settings import VOCAB_SIZE, MAX_LENGTH

class TestKistmatAI(unittest.TestCase):
    def setUp(self):
        self.model = Kistmat_AI(input_shape=(MAX_LENGTH,), output_shape=1)
        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    def test_model_training(self):
        # Generate dummy data
        X_train = np.random.randint(0, VOCAB_SIZE, size=(100, MAX_LENGTH))
        y_train = np.random.rand(100, 1)

        # Train the model
        history = self.model.fit(X_train, y_train, epochs=5, batch_size=10, verbose=0)

        # Check if the model returns a result
        result = self.model.predict(X_train[:1])
        self.assertIsNotNone(result)

        # Check if the model's score is above the threshold
        score = self.model.evaluate(X_train, y_train, verbose=0)
        self.assertGreaterEqual(score[1], 0.8)  # Assuming the second metric is the score

if __name__ == '__main__':
    unittest.main()