import unittest
import torch
import torch.nn as nn
from model import CustomModel  # Make sure to import your LSTM class

class TestConfig:
    "batch_size" : 32 
    "dropout" : 0.2
    "hidden_size" : 128
    "emb_size" : 64
    "n_layers" : 1
    "n_head" : 8
    "seq_len" : 512
    "target_size" : 1

class TestCustomModel(unittest.TestCase):
    def setUp(self):
        self.config = TestConfig()
        self.model = CustomModel(self.config)

    def test_initialization(self):
        # """Test if the model initializes without errors."""
        self.assertIsInstance(self.model, CustomModel)

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