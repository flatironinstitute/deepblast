import torch
from deepblast.embedding import MultiLinear, MultiheadProduct
import unittest


class TestEmbedding(unittest.TestCase):
    def setUp(self):
        b, L, d, h = 3, 100, 50, 8
        self.x = torch.randn(b, L, d)
        self.y = torch.randn(b, L, d)
        self.b = b
        self.L = L
        self.d = d
        self.h = h

    def test_multilinear(self):
        model = MultiLinear(self.d, self.d, self.h)
        res = model(self.x)
        self.assertEqual(tuple(res.shape), (self.b, self.L, self.d, self.h))

    def test_multihead_product(self):
        model = MultiheadProduct(self.d, self.d, self.h)
        res = model(self.x, self.y)
        self.assertEqual(tuple(res.shape), (self.b, self.L, self.L))


if __name__ == '__main__':
    unittest.main()
