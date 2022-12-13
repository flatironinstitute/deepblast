import unittest
from deepblast.utils import get_data_path
from deepblast.dataset import MaliAlignmentDataset, TMAlignDataset
from deepblast.dataset.alphabet import UniprotTokenizer
from deepblast.dataset.utils import collate_f
import pandas as pd
from transformers import T5EncoderModel, T5Tokenizer


class TestTMAlignDataset(unittest.TestCase):
    def setUp(self):
        self.data_path = get_data_path('test_tm_align.tab')
        self.tokenizer = UniprotTokenizer(pad_ends=False)
        self.tokenizer = T5Tokenizer.from_pretrained(
            'Rostlab/prot_t5_xl_uniref50', do_lower_case=False)

    def test_constructor(self):
        x = TMAlignDataset(self.data_path, tm_threshold=0, max_len=10000,
                           tokenizer=self.tokenizer)
        self.assertEqual(len(x), 10)

    def test_getitem(self):
        x = TMAlignDataset(self.data_path, tm_threshold=0,
                           pad_ends=False, clip_ends=True,
                           tokenizer=self.tokenizer)
        res = x[0]
        self.assertEqual(len(res), 8)
        gene, pos, states, alignment_matrix, _, _, _, _ = res
        # test the lengths
        self.assertEqual(len(gene), 21)
        self.assertEqual(len(pos), 21)
        self.assertEqual(len(states), 21)
        # wtf is going on here??
        self.assertEqual(alignment_matrix.shape, (21, 21))

    def test_collate(self):
        x = TMAlignDataset(self.data_path, tm_threshold=0,
                           pad_ends=False, clip_ends=True,
                           tokenizer=self.tokenizer)
        batch = (x[0], x[1], x[2])
        res = collate_f(batch)
        genes, others, states, dm, p, G, gM, oM = res

    def test_gappy_getitem(self):
        TMAlignDataset(self.data_path, tm_threshold=0,
                       pad_ends=False, clip_ends=False,
                       tokenizer=self.tokenizer)
        # TODO: we need to make sure that the ends can be appropriately handled


class TestMaliDataset(unittest.TestCase):

    def setUp(self):
        self.data_path = get_data_path('example.txt')
        self.pairs = pd.read_table(self.data_path, header=None)

    def test_constructor(self):
        x = MaliAlignmentDataset(self.pairs)
        self.assertEqual(len(x), 3)

    def test_getitem(self):
        x = MaliAlignmentDataset(self.pairs)
        res = x[0]
        self.assertEqual(len(res), 4)
        gene, pos, states, alignment_matrix = res
        # test the lengths
        self.assertEqual(len(gene), 81)
        self.assertEqual(len(pos), 81)
        self.assertEqual(len(states), 100)
        self.assertEqual(alignment_matrix.shape, (81, 82))


if __name__ == '__main__':
    unittest.main()
