import os
import shutil
import unittest
from deepblast.trainer import LightningAligner
from deepblast.utils import get_data_path
from pytorch_lightning import Trainer


class TestTrainer(unittest.TestCase):

    def setUp(self):
        # Fake parse args
        self.output_dir = 'output-dir'
        args = [
            '--train-pairs', get_data_path('train.txt'),
            '--test-pairs', get_data_path('test.txt'),
            '--valid-pairs', get_data_path('valid.txt'),
            '--output-directory', self.output_dir,
            '--epochs', '1'
            '--batch-size', '1'
        ]
        parser = LightningAligner.add_model_specific_args(
            parser)
        args = parse_parse_args(args)
        self.model = LightningAligner(args)
        # profiler = AdvancedProfiler()

        self.trainer = Trainer(
            max_nb_epochs=1,
            # profiler=profiler,
            # fast_dev_run=True
            #auto_scale_batch_size='power'
        )

        pass

    def tearDown(self):
        shutil.rmtree(self.output_dir)
        pass

    def test_run(self):
        # Smoke test to make sure that it runs.
        trainer.fit(model)


if __name__ == '__main__':
    unittest.main()
