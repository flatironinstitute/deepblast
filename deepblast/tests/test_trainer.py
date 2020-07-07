import os
import shutil
import unittest
from deepblast.trainer import LightningAligner
from deepblast.utils import get_data_path
from deepblast.sim import hmm_alignments
from pytorch_lightning import Trainer
import argparse


class TestTrainer(unittest.TestCase):
    def setUp(self):
        hmm = get_data_path('ABC_tran.hmm')
        n_alignments = 100
        align_df = hmm_alignments(
            n=40, seed=0, n_alignments=n_alignments, hmmfile=hmm)
        cols = [
            'chain1_name', 'chain2_name', 'tmscore1', 'tmscore2', 'rmsd',
            'chain1', 'chain2', 'alignment'
        ]
        align_df.columns = cols
        parts = n_alignments // 10
        train_df = align_df.iloc[:parts * 8]
        test_df = align_df.iloc[parts * 8:parts * 9]
        valid_df = align_df.iloc[parts * 9:]

        # save the files to disk.
        train_df.to_csv('train.txt', sep='\t', index=None, header=None)
        test_df.to_csv('test.txt', sep='\t', index=None, header=None)
        valid_df.to_csv('valid.txt', sep='\t', index=None, header=None)

    def tearDown(self):
        if os.path.exists('lightning_logs'):
            shutil.rmtree('lightning_logs')
        if os.path.exists('train.txt'):
            os.remove('train.txt')
        if os.path.exists('test.txt'):
            os.remove('test.txt')
        if os.path.exists('valid.txt'):
          os.remove('valid.txt')

    def test_trainer(self):

        output_dir = 'output'
        args = [
            '--train-pairs', 'train.txt',
            '--test-pairs', 'test.txt',
            '--valid-pairs', 'valid.txt',
            '--output-directory', output_dir,
            '--epochs', '1',
            '--batch-size', '3',
            '--num-workers', '4',
            '--learning-rate', '1e-4',
            '--clip-ends', 'False',
            '--visualization-fraction', '0.5',
            '--gpus', '1'
        ]
        parser = argparse.ArgumentParser(add_help=False)
        parser = LightningAligner.add_model_specific_args(parser)
        parser.add_argument('--num-workers', type=int)
        parser.add_argument('--gpus', type=int)
        args = parser.parse_args(args)
        model = LightningAligner(args)
        trainer = Trainer(
            max_epochs=args.epochs,
            gpus=args.gpus,
            check_val_every_n_epoch=1,
            # profiler=profiler,
            fast_dev_run=True,
            # auto_scale_batch_size='power'
        )
        trainer.fit(model)


if __name__ == '__main__':
    unittest.main()
