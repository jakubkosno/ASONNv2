import unittest
from asonn import Asonn
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

class TestAGDS(unittest.TestCase):
    agds = Asonn()

    def test_unique_id(self):
        self.agds.build_agds_from("datasets/iris.txt")
        ids = list()
        for neuron in self.agds.neurons:
            ids.append(neuron.id)

        self.assertEqual(len(ids), len(set(ids)), "IDs are unique")

class TestAGDSWithWeights(unittest.TestCase):
    def get_neuron(self, asonn, value, type):
        try:
            for neuron in asonn.neurons:
                if str(neuron.value) == str(value) and neuron.is_of_type(type):
                    return neuron

            return None
        except:
            for neuron in asonn.neurons:
                if str(neuron.value) == str(value) and neuron.is_of_type_with_weight(type):
                    return neuron

            return None

    def get_adef_connection_weights_to(self, asonn, value, type):
        adef_weights = list()
        value_neuron = self.get_neuron(asonn, value, type)
        if len(value_neuron.connections) == 0:
            self.assertTrue(False, "No such neuron")
        else:
            connected_objects = [x for x in value_neuron.connections if x['neuron'].is_object()]
            for object in connected_objects:
                for neuron in object['neuron'].connections:
                    if neuron['neuron'].id == value_neuron.id:
                        adef_weights.append(neuron['weight'])

        return adef_weights

    def test_weighted_connections(self):
        asonn = Asonn()
        asonn.build_agds_from("datasets/iris.txt")
        asonn.add_weighted_connections()
        for neuron in asonn.neurons:
            for connection in neuron.connections:
                self.assertGreaterEqual(connection['weight'], 0, "All weights bigger than 0")
                self.assertLessEqual(connection['weight'], 1, "All weights smaller than 1")

    def test_sorted_connections(self):
        asonn = Asonn()
        asonn.build_agds_from("datasets/iris.txt")
        asonn.add_weighted_connections()
        for neuron in asonn.neurons:
            if neuron.is_label:
                for i in range(len(neuron.connections) - 1):
                    self.assertLessEqual(neuron.connections[i]["neuron"].value, neuron.connections[i +1]["neuron"].value, "Value nodes are sorted")

    def test_same_type_weights(self):
        asonn = Asonn()
        asonn.build_agds_from("datasets/iris.txt")
        asonn.add_weighted_connections()
        for neuron in asonn.neurons:
            if neuron.is_value() and (type(neuron.value) is float or type(neuron.value is int)):
                has_same_type_connection = False
                for connection in neuron.connections:
                    if connection['neuron'].is_of_type_with_weight(neuron.get_type()):
                        has_same_type_connection = True

                self.assertTrue(has_same_type_connection, "Same type connections added")

    def test_ADEF_connections(self):
        asonn = Asonn()
        asonn.build_agds_from("datasets/iris.txt")
        asonn.add_weighted_connections()
        adef_weights = self.get_adef_connection_weights_to(asonn, 4.9, 'sepal length')
        self.assertTrue(0.21621621621621623 in adef_weights, "Weight calculated correctly")
        self.assertEqual(len(adef_weights), 6, "All connections present")
        adef_weights = self.get_adef_connection_weights_to(asonn, 3.4, 'sepal width')
        self.assertTrue(0.2 in adef_weights, "Weight calculated correctly")
        self.assertEqual(len(adef_weights), 12, "All connections present")


if __name__ == "__main__":
    unittest.main()
