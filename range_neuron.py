import math
from neuron import Neuron


class RangeNeuron(Neuron):
    gamma_p = 1
    exponent = 2

    def __init__(self, neuron, weight):
        super().__init__(neuron.value, False)
        self.value = [self.value]
        neuron.add_connection_with_weight(weight, self)
        self.connections.append({"weight": weight, "neuron": neuron})

    def is_range(self):
        return True

    def is_modified(self):
        return False

    def is_value(self):
        return False

    def has_within_range(self, neuron):
        if neuron.get_type() != self.find_label_connection().value:
            return False

        return neuron.get_type() == self.get_type() and neuron.value > min(self.value) and neuron.value < max(self.value)

    def get_pattern_connection(self):
        for connection in self.connections:
            if connection['neuron'].is_pattern():
                return connection['neuron']

    def find_label_connection(self):
        for connection in self.connections:
            if connection['neuron'].is_value():
                return connection['neuron'].get_label_connection()

    def get_feature_type(self):
        for connection in self.connections:
            if connection['neuron'].is_value():
                return connection['neuron'].get_type()

    def get_type(self):
        for connection in self.connections:
            if connection['neuron'].is_pattern():
                return connection['neuron'].get_type()

    def get_out_objects_correlated(self):
        correlated = list()
        for connection in self.get_value_connections():
            correlated.extend([x.value for x in connection.get_object_connections() if x.get_object_type() != self.get_type()])

        return correlated

    def get_in_objects_correlated(self, exceptions):
        correlated = list()
        for connection in self.get_value_connections():
            correlated.extend([x.value for x in connection.get_object_connections() if x.get_object_type() == self.get_type() and x not in exceptions])

        return correlated

    def get_possible_extensions(self, class_type):
        possible_expansions = list()
        label_neuron = self.find_label_connection()
        minus = None
        plus = None

        for connection in label_neuron.connections:
            if connection['neuron'].value == max(self.value):
                for potential_connection in connection['neuron'].get_asim_connections():
                    if potential_connection.value not in self.value and potential_connection.value > max(self.value):
                        plus = potential_connection

        for connection in label_neuron.connections:
            if connection['neuron'].value == min(self.value):
                for potential_connection in connection['neuron'].get_asim_connections():
                    if potential_connection.value not in self.value and potential_connection.value < min(self.value):
                        minus = potential_connection


        minus_objects = list()
        plus_objects = list()
        for connection in label_neuron.connections:
            if not minus is None:
                if connection['neuron'].value <= minus.value:
                    minus_objects.extend([x.get_object_type() for x in connection['neuron'].get_object_connections()])
                    continue
            if not plus is None:
                if connection['neuron'].value >= plus.value:
                    plus_objects.extend([x.get_object_type() for x in connection['neuron'].get_object_connections()])

        if class_type in plus_objects:
            possible_expansions.append(plus)

        if class_type in minus_objects:
            possible_expansions.append(minus)

        return possible_expansions

    def extend(self, neuron):
        self.connections.append({'weight': 1, 'neuron': neuron})
        neuron.add_connection_with_weight(1, self)
        self.value.append(neuron.value)
        _ = self.get_possible_extensions(self.get_type())

    def remove_connection(self, neuron):
        self.connections = [x for x in self.connections if x['neuron'] != neuron]
        neuron.remove_connection(self)
        if neuron.value in self.value:
            self.value.remove(neuron.value)

    def calculate_minus_coefficient(self, smaller_values): #all value neurons from same category and value smaller than min in range
        #7.31
        return sum([self.calculate_7_16(value_neuron) for value_neuron in smaller_values]) / self.gamma_p

    def calculate_plus_coefficient(self, bigger_values):
        #7.32
        return sum([self.calculate_7_17(value_neuron) for value_neuron in bigger_values]) / self.gamma_p

    def calculate_7_16(self, value_neuron):
        #7.16
        return self.calculate_7_18(value_neuron) * \
        (sum([self.calculate_7_20(object_neuron) * self.calculate_7_21(object_neuron) for object_neuron in value_neuron.get_object_connections() if object_neuron.get_object_type() == self.get_type()]) - \
        sum([self.calculate_7_21(object_neuron) for object_neuron in value_neuron.get_object_connections() if object_neuron.get_object_type() != self.get_type()]))

    def calculate_7_17(self, value_neuron):
        #7.17
        return self.calculate_7_19(value_neuron) * \
        (sum([self.calculate_7_20(object_neuron) * self.calculate_7_21(object_neuron) for object_neuron in value_neuron.get_object_connections() if object_neuron.get_object_type() == self.get_type()]) - \
        sum([self.calculate_7_21(object_neuron) for object_neuron in value_neuron.get_object_connections() if object_neuron.get_object_type() != self.get_type()]))

    def calculate_7_18(self, value_neuron):
        #7.18
        return pow((1 - (min(self.value) - value_neuron.value) / self.find_label_connection().get_range()), 2)

    def calculate_7_19(self, value_neuron):
        #7.19
        return pow((1 - (max(self.value) - value_neuron.value) / self.find_label_connection().get_range()), 2)

    def get_in_objects_from(self, values):
        in_objects = list()
        for value_neuron in values:
            for object_neuron in value_neuron.get_object_connections():
                if object_neuron.get_object_type() == self.get_type():
                    in_objects.append(object_neuron)

        return in_objects

    def get_out_objects_from(self, values):
        out_objects = list()
        for value_neuron in values:
            for object_neuron in value_neuron.get_object_connections():
                if object_neuron.get_object_type() != self.get_type():
                    out_objects.append(object_neuron)

        return out_objects

    def calculate_7_20(self, object_neuron):
        #7.20
        return pow((1 / (1 + self.calculate_7_24(object_neuron))), self.exponent)

    def calculate_7_21(self, object_neuron):
        #7.21
        result = pow(((1 + self.calculate_7_25(object_neuron))/ len(self.get_pattern_connection().get_range_connections())), 1)
        return result

    def calculate_7_24(self, object_neuron):
        #7.24
        return object_neuron.count_pattern_connections()

    def calculate_7_25(self, object_neuron):
        #7.25
        return self.calculate_7_26(object_neuron) + self.calculate_7_27(object_neuron)

    def calculate_7_26(self, object_neuron):
        #7.27
        return len([x for x in object_neuron.get_value_connections() if self.has_within_range(x)])

    def calculate_7_27(self, object_neuron):
        #7.27
        return 0

    def get_gaussian_hat_value(self, value):
        #7.40
        if value < min(self.value) or value > max(self.value):
            if max(self.value) == min(self.value):
                return 0

            return pow(math.e, (1 - pow(((2 * value - max(self.value) - min(self.value)) / (max(self.value) - min(self.value))), 2)) / 2)
        else:
            return 1

    def set_adef_weight(self, weight):
        for connection in self.connections:
            if connection['neuron'].is_pattern():
                connection['weight'] = weight

    def calculate_7_34(self, asonn):
        #7.34
        return (1 - self.count_weeds() / self.calculate_7_37(asonn)) * (self.calculate_7_36(asonn) + self.count_seeds()) / (self.calculate_7_36(asonn) + self.get_pattern_connection().count_seeds())

    def calculate_7_36(self, asonn):
        #7.36
        return self.get_quantity_of_objects_represented_by_pattern() * pow(asonn.get_attribute_quantity(), 2)

    def calculate_7_37(self, asonn):
        #7.37
        return len([x for x in asonn.neurons if not x.is_of_type_with_weight(self.get_type())]) * pow(asonn.get_attribute_quantity(), 2)

    def get_quantity_of_objects_represented_by_pattern(self):
        for connection in self.connections:
            if connection['neuron'].is_pattern():
                return connection['neuron'].get_represented_objects_quantity()

    def count_seeds(self):
        counter = 0
        for connection in self.connections:
            if connection['neuron'].is_value():
                counter += len([x for x in connection['neuron'].connections if x['neuron'].is_object() and x['neuron'].is_of_type_with_weight(self.get_type())])

        return counter

    def count_weeds(self):
        counter = 0
        for connection in self.connections:
            if connection['neuron'].is_value():
                counter += len([x for x in connection['neuron'].connections if x['neuron'].is_object() and not x['neuron'].is_of_type_with_weight(self.get_type())])

        return counter

    def get_seeds(self):
        seeds = list()
        for connection in self.connections:
            if connection['neuron'].is_value():
                seeds += [x['neuron'].value for x in connection['neuron'].connections if x['neuron'].is_object() and not x['neuron'].is_of_type_with_weight(self.get_type())]

        return seeds

    def activate(self, value):
        for connection in self.connections:
            if connection['neuron'].is_pattern():
                connection['neuron'].activate(self.get_gaussian_hat_value(value) * connection['weight'])

    def reduce(self):
        min_val = min(self.value)
        max_val = max(self.value)
        self.value = [min_val, max_val]
