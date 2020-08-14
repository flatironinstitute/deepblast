import numpy as np
import unittest
from deepblast.dataset.alphabet import UniprotTokenizer
import numpy.testing as npt


class TestAlphabet(unittest.TestCase):

    def test_tokenizer(self):
        tokenizer = UniprotTokenizer(pad_ends=True)
        res = tokenizer(b'ARNDCQEGHILKMFPSTWYVXOUBZ')
        # Need to account for padding and offset
        exp = np.array([20] + list(range(0, 21)) + [11, 4, 20, 20] + [20])
        npt.assert_allclose(res, exp)

    def test_tokenizer_encode(self):
        tokenizer = UniprotTokenizer(pad_ends=True)
        x = 'ARNDCQEGHILKMFPSTWYVXOUBZ'
        x = str.encode(x)
        res = tokenizer(x)
        exp = np.array(
            [20, 0, 1, 2, 3, 4, 5, 6, 7, 8,
             9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
             19, 20, 11, 4, 20, 20, 20])
        npt.assert_allclose(exp, res)

    def test_tokenizer_encode_no_padding(self):
        tokenizer = UniprotTokenizer(pad_ends=False)
        x = 'ARNDCQEGHILKMFPSTWYVXOUBZ'
        x = str.encode(x)
        res = tokenizer(x)
        exp = np.array(
            [0, 1, 2, 3, 4, 5, 6, 7, 8,
             9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
             19, 20, 11, 4, 20, 20])
        npt.assert_allclose(exp, res)


if __name__ == '__main__':
    unittest.main()
