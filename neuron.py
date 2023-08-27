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
        self.out_correlation = list()

    def is_of_type(self, label):
        if self.is_label:
            return self.value == label
        for neuron in self.connections:
            if str(neuron.value) == str(label):
                return True
        return False

    def add_connection(self, neuron):
        self.connections.append(neuron)

    def add_connection_with_weight(self, weight, neuron):
        self.connections.append({'weight': weight, 'neuron': neuron})

    def __eq__(self, other):
        return self.id == other.id

    def __hash__(self):
        return hash(self.id)

    def remove_duplicated_connections(self):
        self.connections = list(set(self.connections))

    def replace_connections(self, new_connections):
        self.connections = new_connections

    def sort_connections(self):
        if self.is_label:
            self.connections.sort(key=lambda x: x.value, reverse=False)

    def is_object(self):
        if type(self.value) is str:
            if self.value[0] == "O":
                try:
                    int(self.value[1:])
                    return True
                except:
                    return False
            else:
                return False
        else:
            return False

    def is_value(self):
        return not self.is_label and not self.is_object()

    def get_type(self):
        if self.is_label:
            return self.value
        else:
            try:
                for neuron in self.connections:
                    if neuron.is_label:
                        return neuron.value
            except:
                for neuron in self.connections:
                    if neuron['neuron'].is_label:
                        return neuron['neuron'].value

    def get_class(self):
        if not self.is_object():
            return ""

        object_class = [x['neuron'].value for x in self.connections if x['neuron'].is_of_type_with_weight("Class")]
        return object_class[0]

    def is_of_type_with_weight(self, label):
        if self.is_label:
            return self.value == label

        for connection in self.connections:
            if str(connection['neuron'].value) == str(label):
                return True
        return False

    def calculate_adef_weight(self):
        if not self.is_object():
            return

        value_connections = [x for x in self.connections if x['neuron'].is_value() and x['neuron'].get_type() != "Class"]
        denominator = 0
        for neuron in value_connections:
            denominator += len([x for x in neuron['neuron'].connections if x['neuron'].is_object() and x['neuron'].get_class() == self.get_class()]) / len([x for x in neuron['neuron'].connections if x['neuron'].is_object()])

        for neuron in value_connections:
            one_value_same_adef = len([x for x in neuron['neuron'].connections if x['neuron'].is_object() and x['neuron'].get_class() == self.get_class()])
            one_value_adef = len([x for x in neuron['neuron'].connections if x['neuron'].is_object()])
            neuron['weight'] = ((one_value_same_adef / one_value_adef) / denominator)

    def get_value_connections(self):
        return [x['neuron'] for x in self.connections if x['neuron'].is_value() and x['neuron'].get_type() != "Class"]

    def get_value_connections_with_weights(self):
        return [x for x in self.connections if x['neuron'].is_value() and x['neuron'].get_type() != "Class"]

    def get_object_connections(self):
        return [x['neuron'] for x in self.connections if x['neuron'].is_object()]

    def get_out_correlation(self):
        return self.out_correlation

    def has_bigger_out_correlation(self, other):
        if len(self.out_correlation) == len(other.get_out_correlation()):
            for i in range(len(self.out_correlation)):
                if self.out_correlation[-1 * i] > other.get_out_correlation()[-1 * i]:
                    return True
                elif self.out_correlation[-1 * i] < other.get_out_correlation()[-1 * i]:
                    return False
                else:
                    continue

            return True #True if both out_correlations equal

        else:
            return len(self.out_correlation) > len(other.get_out_correlation())

    def is_pattern(self):
        return False

    def is_range(self):
        return False

    def get_label_connection(self):
        labels = [x['neuron'] for x in self.connections if x['neuron'].is_label]
        return labels[0]

    def get_asim_connections(self):
        return [x['neuron'] for x in self.connections if x['neuron'].is_value() and x['neuron'].get_type() == self.get_type()]

    def get_object_type(self):
        if not self.is_object():
            return None
        else:
            for connection in self.connections:
                if connection['neuron'].get_type() == "Class":
                    return connection['neuron'].value

    def get_range(self):
        if self.is_label:
            values = [x.value for x in self.get_value_connections()]
            return max(values) - min(values)
        else:
            return 0

    def count_pattern_connections(self):
        return len([x for x in self.connections if x['neuron'].is_pattern()])
