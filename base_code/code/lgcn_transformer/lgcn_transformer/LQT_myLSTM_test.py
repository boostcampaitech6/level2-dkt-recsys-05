import unittest
import torch
import torch.nn as nn
from model import LSTM  # Make sure to import your LSTM class

class TestConfig:
    hidden_size = 10
    n_layers = 2

class TestLSTM(unittest.TestCase):
    def setUp(self):
        self.config = TestConfig()
        self.model = LSTM(self.config)

    def test_initialization(self):
        # """Test if the model initializes without errors."""
        self.assertIsInstance(self.model, LSTM)

    def test_forward_pass(self):
        # """Test the forward pass with different input sizes."""
        test_input = torch.randn(5, 3, self.config.hidden_size)  # Batch size 5, Seq len 3
        output = self.model(test_input)
        self.assertEqual(output.shape, (5, 3))  # Check output shape

        test_input = torch.randn(1, 10, self.config.hidden_size)  # Batch size 1, Seq len 10
        output = self.model(test_input)
        self.assertEqual(output.shape, (1, 10))  # Check output shape

    # Add more tests as needed

if __name__ == '__main__':
    unittest.main()