import torch
from deepblast.embedding import MultiLinear, MultiheadProduct
import unittest


class TestEmbedding(unittest.TestCase):
    def setUp(self):
        b, l, d, h = 3, 100, 50, 8
        self.x = torch.randn(b, l, d)
        self.y = torch.randn(b, l, d)
        self.b = b
        self.l = l
        self.d = d
        self.h = h

    def test_multilinear(self):
        model = MultiLinear(self.d, self.d, self.h)
        res = model(self.x)
        self.assertEqual(tuple(res.shape), (self.b, self.l, self.d, self.h))

    def test_multihead_product(self):
        model = MultiheadProduct(self.d, self.d, self.h)
        res = model(self.x, self.y)
        self.assertEqual(tuple(res.shape), (self.b, self.l, self.l))


if __name__ == '__main__':
    unittest.main()
