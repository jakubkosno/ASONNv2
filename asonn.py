from dataset_loader import DatasetLoader
from neuron import Neuron


def format_record(record, labels):
        formatted = list()
        for i in range(len(record)):
            formatted.append((labels[i], record[i]))

        return formatted

class Asonn:
    def __init__(self) -> None:
        self.neurons = set()
    
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

    def __insert(self, record, record_number):
        object_neuron = Neuron("O" + str(record_number))
        value_neurons = set()
        for type, value in record:
            neuron = self.__get_neuron(value, type.value)
            neuron.add_connection(object_neuron)
            object_neuron.add_connection(neuron)
            for label_neuron in self.neurons:
                if label_neuron.is_of_type(type.value) and label_neuron.is_label:
                    neuron.add_connection(label_neuron)
                    label_neuron.add_connection(neuron)

            if neuron not in self.neurons:
                value_neurons.add(neuron)

        self.neurons.add(object_neuron)
        self.neurons.update(value_neurons)

    def __get_neuron(self, value, label):
        for neuron in self.neurons:
            if str(neuron.value) == str(value) and neuron.is_of_type(label):
                return neuron

        return Neuron(value)

    def __remove_duplicated_connections(self):
        for neuron in self.neurons:
            neuron.remove_duplicated_connections()

