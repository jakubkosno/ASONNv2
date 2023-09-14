import csv
import numpy
import pmlb
from scipy.special import softmax
from sklearn.model_selection import train_test_split
from statistics import mode
from dataset_loader import DatasetLoader
from neuron import Neuron
from pattern_neuron import PatternNeuron
from range_neuron import RangeNeuron


def format_record(record, labels):
        formatted = list()
        for i in range(len(record)):
            formatted.append((labels[i], record[i]))

        return formatted

def read_csv(dataset):
    filename_test  = 'datasets/' + dataset + '/' + dataset + '_test.csv'
    filename_train  = 'datasets/' + dataset + '/' + dataset + '_train.csv'
    counter = 0
    X_test = numpy.array([])
    y_test = numpy.array([])
    with open(filename_train, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if counter == 0:
                counter += 1
                continue
            elif counter == 1:
                X_train = numpy.array(numpy.asarray([row[:-1]], dtype=float))
                y_train = numpy.array(numpy.asarray([row[-1]], dtype=float))
            else:
                X_train = numpy.vstack([X_train, row[:-1]])
                y_train = numpy.vstack([y_train, row[-1]])

            counter += 1
    counter = 0
    with open(filename_test, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if counter == 0:
                counter += 1
                continue
            elif counter == 1:
                X_test = numpy.array(numpy.asarray([row[:-1]], dtype=float))
                y_test = numpy.array(numpy.asarray([row[-1]], dtype=float))
            else:
                X_test = numpy.vstack([X_train, row[:-1]])
                y_test = numpy.vstack([y_train, row[-1]])

            counter += 1

    return X_train.astype(numpy.float_), y_train.astype(numpy.float_), X_test.astype(numpy.float_), y_test.astype(numpy.float_)


class Asonn:
    def __init__(self) -> None:
        self.neurons = set()
        self.object_neurons = set()
        self.pattern_neurons = set()
        self.range_neurons = set()

    #AGDS
    def build_agds_from(self, file):
        dataset_loader = DatasetLoader()
        data = dataset_loader.load_from_file(file)
        labels = list()
        for i in range(len(data[0])):
            label_neuron = Neuron(data[0][i], True)
            labels.append(label_neuron)
            self.neurons.add(label_neuron)

        for i in range(len(data) - 1):
            self.__insert(format_record(data[i+1], labels), i+1)

        self.__remove_duplicated_connections()

    def build_agds(self, features, labels):
        label_neurons = list()
        class_neuron = Neuron('Class', True)
        label_neurons.append(class_neuron)
        self.neurons.add(class_neuron)
        for i in range(len(features[0])):
            label_neuron = Neuron('label ' + str(i), True)
            label_neurons.append(label_neuron)
            self.neurons.add(label_neuron)

        for i in range(len(labels)):
            self.__insert(format_record(numpy.append(labels[i], features[i]), label_neurons), i+1)

        self.__remove_duplicated_connections()

    def __insert(self, record, record_number):
        object_neuron = Neuron("O" + str(record_number))
        value_neurons = set()
        for type, value in record:
            neuron = self.__get_neuron(value, type.value)
            neuron.add_connection(object_neuron)
            object_neuron.add_connection(neuron)
            for item in self.neurons:
                if item.is_of_type(type.value) and item.is_label:
                    neuron.add_connection(item)
                    item.add_connection(neuron)

            if neuron not in self.neurons:
                value_neurons.add(neuron)

        self.object_neurons.add(object_neuron)
        self.neurons.update(value_neurons)

    def __get_neuron(self, value, label):
        try:
            for neuron in self.neurons:
                if str(neuron.value) == str(value) and neuron.is_of_type(label):
                    return neuron

            return Neuron(value)
        except:
            for neuron in self.neurons:
                if str(neuron.value) == str(value) and neuron.is_of_type_with_weight(label):
                    return neuron

            return Neuron(value)

    def __remove_duplicated_connections(self):
        for neuron in self.neurons:
            neuron.remove_duplicated_connections()

    def add_weighted_connections(self):
        for neuron in self.neurons:
            neuron.sort_connections()
            connections_with_weights = list()
            for connection in neuron.connections:
                connections_with_weights.append({"weight": 1, "neuron": connection})

            neuron.replace_connections(connections_with_weights)

        for neuron in self.object_neurons:
            connections_with_weights = list()
            for connection in neuron.connections:
                connections_with_weights.append({"weight": 1, "neuron": connection})

            neuron.replace_connections(connections_with_weights)

        self.__add_asim_connections()
        self.__update_adef_weights()

    def __add_asim_connections(self):
        for neuron in self.neurons:
            if neuron.is_label and neuron.value != "Class":
                for i in range(len(neuron.connections) - 1):
                    value_range = neuron.connections[len(neuron.connections) -1]["neuron"].value - neuron.connections[0]["neuron"].value
                    weight = pow(((value_range - (neuron.connections[i + 1]["neuron"].value - neuron.connections[i]["neuron"].value)) / value_range), 1)
                    neuron.connections[i]["neuron"].add_connection_with_weight(weight, neuron.connections[i + 1]['neuron'])
                    neuron.connections[i + 1]["neuron"].add_connection_with_weight(weight, neuron.connections[i]['neuron'])


    def __update_adef_weights(self):
        for neuron in self.object_neurons:
            neuron.calculate_adef_weight()

    #ASONN
    def build_asonn(self):
        self.calculate_out_correlations()
        not_represented = self.object_neurons
        while len(not_represented) > 0:
            pattern_neuron = self.add_pattern(not_represented)
            self.reduce_ranges()
            self.expand_pattern(pattern_neuron)
            not_represented = [x for x in self.object_neurons if x not in self.get_represented_objects()]

        self.add_asonn_adef_connections()
        self.reduce_ranges()

    def calculate_out_correlations(self):
        for object_neuron in self.object_neurons:
            bad_correlations = list()
            value_connections = object_neuron.get_value_connections()
            for value_neuron in value_connections:
                object_connections = value_neuron.get_object_connections()
                bad_correlations += [x for x in object_connections if x.get_class() != object_neuron.get_class()]

            bad_correlations_summed = list()
            while len(bad_correlations):
                no_duplicates = set(bad_correlations)
                bad_correlations_summed.append(len(no_duplicates))
                for value in no_duplicates:
                    bad_correlations.remove(value)

            bad_correlations_summed.append(0)
            for i in range(len(bad_correlations_summed) - 1):
                object_neuron.get_out_correlation().append(bad_correlations_summed[i] - bad_correlations_summed[i + 1])
    def reduce_ranges(self):
        for neuron in self.range_neurons:
            neuron.reduce()

    def find_biggest_out_correlation(self, object_neurons):
        biggest_out_corr = list(object_neurons)[0]
        for neuron in object_neurons:
            if neuron.is_object() and neuron.has_bigger_out_correlation(biggest_out_corr):
                biggest_out_corr = neuron

        return biggest_out_corr

    def add_pattern(self, not_represented):
        pattern_neuron = PatternNeuron(self.find_biggest_out_correlation(not_represented))
        self.pattern_neurons.add(pattern_neuron)
        for connection in pattern_neuron.connections:
            if connection['neuron'].is_range():
                self.range_neurons.add(connection['neuron'])

        return pattern_neuron

    def expand_pattern(self, pattern_neuron):
        pattern_neuron.expand(self.neurons)

    def get_represented_objects(self):
        represented = list()
        for pattern_neuron in self.pattern_neurons:
            represented.extend(pattern_neuron.get_object_connections())

        return represented

    def add_asonn_adef_connections(self):
        for neuron in self.pattern_neurons:
            neuron.add_adef_weights(self)

    def get_attribute_quantity(self):
        return len([x for x in self.neurons if x.is_label])

    def build(self, features, labels):
        self.build_agds(features, labels)
        self.add_weighted_connections()
        self.build_asonn()

    def classify(self, features):
        self.reset_activations()
        for i in range(len(features)):
            self.activate(features[i], i)

        pattern_neurons = list(self.pattern_neurons)
        softmax_activations = softmax([x.get_activation() for x in pattern_neurons])
        max_activation = 0
        max_activated_neuron = list()
        for i in range(len(pattern_neurons)):
            if softmax_activations[i] > max_activation:
                max_activation = softmax_activations[i]
                max_activated_neuron = [pattern_neurons[i]]

            elif softmax_activations[i] == max_activation:
                max_activated_neuron.append(pattern_neurons[i])

        if len(max_activated_neuron) < 2:
            result = max_activated_neuron[0].get_type()
        else:
            result = mode([x.get_type() for x in max_activated_neuron])

        return result

    def activate(self, feature, feature_number):
        for neuron in [x for x in self.range_neurons if x.get_feature_type() == 'label ' + str(feature_number)]:
            neuron.activate(feature)

    def reset_activations(self):
        for neuron in self.pattern_neurons:
            neuron.reset_activation()

    def get_most_activated_neuron(self):
        activations = sorted(self.pattern_neurons, key=lambda x: x.get_activation(), reverse=True)
        return activations[0]

    def score(self, dataset, labels):
        if len(dataset) != len(labels):
            print("Labels don't fit given dataset")
            return 0

        if len(dataset) < 1:
            print("Empty dataset")
            return 0

        correct = 0
        all = 0
        for i in range(len(dataset)):
            if self.classify(dataset[i]) == labels[i]:
                correct += 1

            all += 1

        return correct / all


if __name__ == "__main__":
    asonn = Asonn()
    X_train, y_train, X_test, y_test = read_csv("iris")
    asonn.build(X_train, y_train)
    print("Score: " + str(asonn.score(X_test, y_test)))
