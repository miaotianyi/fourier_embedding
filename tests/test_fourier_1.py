import unittest

import torch
from layers.fourier_features import FourierFeatures


class MyTestCase(unittest.TestCase):
    def test_shape(self):
        h_in, h_out = 7, 20
        ff = FourierFeatures(h_in, h_out, sigma=1., trainable=False)
        x = torch.rand(8, 6, 3, h_in)
        self.assertListEqual(list(ff(x).shape), [8, 6, 3, h_out])

    def test_trainable(self):
        # when the layer is trainable, the output should be different after training
        pass

    def test_not_trainable(self):
        # when the layer is non-trainable, the output should remain the same
        pass


if __name__ == '__main__':
    unittest.main()
