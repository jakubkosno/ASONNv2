import unittest
from dataset_loader import DatasetLoader


class TestDatasetLoader(unittest.TestCase):
    dataset_loader = DatasetLoader()

    def test_wrong_file(self):
        dataset = self.dataset_loader.load_from_file("wrong_file.txt")
        self.assertEqual(len(dataset), 0, "Wrong file is not loaded")

    def test_class_label_added(self):
        dataset = self.dataset_loader.load_from_file("datasets/iris.txt")
        self.assertEqual(dataset[0][0], "Class", "Class label added")

    def test_loading_iris(self):
        iris = self.dataset_loader.load_from_file("datasets/iris.txt")
        self.assertEqual(len(iris), 151, "Dataset size correct")
        self.assertEqual(len(iris[1]), 5, "Record length correct")
        self.assertEqual(iris[1][0], "Iris-setosa", "String value correct")
        self.assertEqual(iris[1][1], 5.1, "Numeric value correct")


if __name__ == "__main__":
    unittest.main()
