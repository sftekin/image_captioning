from unittest import TestCase
from batch_generator import BatchGenerator


class TestBatchGenerator(TestCase):
    def setUp(self):
        pass

    def test_forward_dims(self):
        generator = BatchGenerator(data_path={"image_path": "../dataset/images/",
                                              "caption_path": "../dataset/imid.csv",
                                              "sentence_path": "../dataset/captions.csv"},
                                   batch_size=8,
                                   min_num_captions=4,
                                   im_size=(250, 500))

        images, captions = next(generator)
        self.assertEqual(images.shape, (32, 250, 500, 3))
