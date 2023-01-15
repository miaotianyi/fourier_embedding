import torch
from layers.running_norm import RunningNorm

import unittest


class MyTestCase(unittest.TestCase):
    def test_something(self):
        layer = RunningNorm([-2, -3], [4, 3], momentum=None)
        layer.train()
        for i in range(10):
            a = torch.randn((2, 3, 4, 5)) * 10 + 6
            layer(a)
        print()
        print(layer.running_mean)
        print(layer.running_var)
        layer.eval()
        for i in range(10):
            a = torch.zeros((7, 2, 3, 4, 5))    # negative indices, so different shape is allowed
            layer(a)
        print("after eval on other input")  # should be the same thanks to .eval()
        print(layer.running_mean)
        print(layer.running_var)


if __name__ == '__main__':
    unittest.main()
