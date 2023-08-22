from itertools import count


class Neuron:
    _ids = count(0)

    def __init__(self, value, is_label = False):
        if type(value) == str:
            self.value = value.strip()
        else:
            self.value = value
        self.connections = list()
        self.id = next(self._ids)
        self.is_label = is_label

    def is_of_type(self, label):
        if self.is_label:
            return self.value == label
        for neuron in self.connections:
            if str(neuron.value) == str(label):
                return True
        return False

    def add_connection(self, neuron):
        self.connections.append(neuron)

    def __eq__(self, other):
        return self.id == other.id

    def __hash__(self):
        return hash(self.id)

    def remove_duplicated_connections(self):
        self.connections = list(set(self.connections))
