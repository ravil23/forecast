import unittest
from tensorflow_oop import *

class TestTFDataset(unittest.TestCase):
    def setUp(self):
        self.empty = TFDataset()
        self.data = [[1,2],[3,4],[5,6]]
        self.labels = [1,2,3]
        self.dataset1 = TFDataset(data=self.data)
        self.dataset2 = TFDataset(labels=self.labels)
        self.dataset3 = TFDataset(data=self.data, labels=self.labels)

    def test_init(self):
        self.assertFalse(self.empty.init_)
        self.assertEqual(self.empty.batch_size_, 1)
        self.assertEqual(self.empty.batch_num_, 0)

        self.assertTrue(self.dataset1.init_)
        self.assertEqual(self.dataset1.batch_size_, 1)
        self.assertEqual(self.dataset1.batch_num_, 0)

        self.assertTrue(self.dataset2.init_)
        self.assertEqual(self.dataset2.batch_size_, 1)
        self.assertEqual(self.dataset2.batch_num_, 0)

        self.assertTrue(self.dataset3.init_)
        self.assertEqual(self.dataset3.batch_size_, 1)
        self.assertEqual(self.dataset3.batch_num_, 0)

    def test_data_initialize(self):
        self.empty.initialize(self.data, None)
        self.assertEqual(self.empty.size_, 3)
        self.assertEqual(self.empty.data_shape_, [2])
        self.assertEqual(self.empty.data_ndim_, 1)
        self.assertEqual(self.empty.labels_shape_, None)
        self.assertEqual(self.empty.labels_ndim_, None)
        self.assertEqual(self.empty.batch_count_, 3)
        self.assertTrue(self.empty.init_)

    def test_labels_initialize(self):
        self.empty.initialize(None, self.labels)
        self.assertEqual(self.empty.size_, 3)
        self.assertEqual(self.empty.data_shape_, None)
        self.assertEqual(self.empty.data_ndim_, None)
        self.assertEqual(self.empty.labels_shape_, [1])
        self.assertEqual(self.empty.labels_ndim_, 1)
        self.assertEqual(self.empty.batch_count_, 3)
        self.assertTrue(self.empty.init_)

    def test_data_labels_initialize(self):
        self.empty.initialize(self.data, self.labels)
        self.assertEqual(self.empty.size_, 3)
        self.assertEqual(self.empty.data_shape_, [2])
        self.assertEqual(self.empty.data_ndim_, 1)
        self.assertEqual(self.empty.labels_shape_, [1])
        self.assertEqual(self.empty.labels_ndim_, 1)
        self.assertEqual(self.empty.batch_count_, 3)
        self.assertTrue(self.empty.init_)

    def test_deep_copy(self):
        self.empty.deep_copy(self.dataset3)
        for attr in self.empty.__slots__:
            self.assertTrue(np.all(np.asarray(getattr(self.empty, attr)) == np.asarray(getattr(self.dataset3, attr))))

if __name__ == '__main__':
    unittest.main()
