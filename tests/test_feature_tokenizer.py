import unittest
import torch
from layers.feature_tokenizer import FeatureTokenizer


class MyTestCase(unittest.TestCase):
    def test_shape_1(self):
        n = 10
        num_features = 7
        h_in = 13
        h_out = 8

        with torch.inference_mode():
            x = torch.rand(n, num_features, h_in)
            ft = FeatureTokenizer(num_features, h_in, h_out)
            y = ft(x)
            self.assertEqual(list(y.shape), [n, num_features, h_out])

    def test_shape_2(self):
        # case 2: input numerical features are not embedded
        n = 10
        num_features = 7
        h_in = 1
        h_out = 8

        with torch.inference_mode():
            x = torch.rand(n, num_features)
            ft = FeatureTokenizer(num_features, h_in, h_out)
            y = ft(x)
            self.assertEqual(list(y.shape), [n, num_features, h_out])


if __name__ == '__main__':
    unittest.main()
