from itertools import count
from statistics import mode
from neuron import Neuron
from range_neuron import RangeNeuron


class PatternNeuron(Neuron):
    _pattern_ids = count(1)

    def __init__(self, neuron, max_discrimination=1):
        self.max_discrimination = max_discrimination
        value = "P" + str(next(self._pattern_ids))
        super().__init__(value, False)
        for connection in neuron.connections:
            if connection['neuron'].get_type() == "Class":
                connection['neuron'].add_connection_with_weight(1, self)
                self.add_connection_with_weight(1, connection['neuron'])
        for connection in neuron.get_value_connections_with_weights():
            range_neuron = RangeNeuron(connection['neuron'], connection['weight'])
            range_neuron.add_connection_with_weight(1, self)
            self.connections.append({'weight': 1, 'neuron': range_neuron})
        self.add_connection_with_weight(1, neuron)
        neuron.add_connection_with_weight(1, self)
        self.activation = 0

    def is_value(self):
        return False

    def is_modified(self):
        return False

    def is_pattern(self):
        return True

    def get_type(self):
        for connection in self.connections:
            if connection['neuron'].get_type() == "Class":
                return connection['neuron'].value

    def get_range_connections(self):
        return [x['neuron'] for x in self.connections if x['neuron'].is_range()]

    def activate(self, value):
        self.activation += value

    def reset_activation(self):
        self.activation = 0

    def get_activation(self):
        return self.activation

    def expand(self, all_neurons):
        case_one_not_done = True
        case_two_not_done = True
        while case_one_not_done or case_two_not_done:
            expansion_options = list()
            for connection in self.connections:
                if connection['neuron'].is_range():
                    expansion_options.extend(connection['neuron'].get_possible_extensions(self.get_type()))

            case_one_not_done = self.expand_with_one_class_values(expansion_options)
            case_two_not_done = self.try_best_extension(all_neurons)
            self.add_new_object_connections_if_represented()

    def expand_step(self, all_neurons):
        case_one_not_done = True
        case_two_not_done = True
        expansion_options = list()
        for connection in self.connections:
            if connection['neuron'].is_range():
                expansion_options.extend(connection['neuron'].get_possible_extensions(self.get_type()))

        case_one_not_done = self.expand_with_one_class_values(expansion_options)
        case_two_not_done = self.try_best_extension(all_neurons)
        self.add_new_object_connections_if_represented()
        return case_one_not_done, case_two_not_done


    def expand_with_one_class_values(self, expansion_options):
        for new_value_neuron in expansion_options:
            if len([x for x in new_value_neuron.get_object_connections() if x.get_object_type() != self.get_type()]) == 0:
                for range_neuron in [x for x in self.connections if x['neuron'].is_range() and x['neuron'].find_label_connection().value == new_value_neuron.get_type()]:
                    range_neuron['neuron'].extend(new_value_neuron)
                    return True

        return False

    def get_current_discrimination(self):
        all_correlated_out = list()
        for range_neuron in self.get_range_connections():
            all_correlated_out.extend(range_neuron.get_out_objects_correlated())

        if len(all_correlated_out) == 0:
            return len(self.get_range_connections())
        else:
            return len(self.get_range_connections()) - all_correlated_out.count(mode(all_correlated_out))

    def try_best_extension(self, all_neurons):
        extensions = list()
        for range_neuron in self.get_range_connections():
            for new_neuron in range_neuron.get_possible_extensions(self.get_type()):
                if new_neuron.value > max(range_neuron.value):
                    coeff = range_neuron.calculate_plus_coefficient([x for x in all_neurons if x.is_value() and x.get_type() == range_neuron.find_label_connection().value and x.value > max(range_neuron.value)])
                    if coeff > 0:
                        extensions.append([coeff, new_neuron, range_neuron])
                elif new_neuron.value < min(range_neuron.value):
                    coeff = range_neuron.calculate_minus_coefficient([x for x in all_neurons if x.is_value() and x.get_type() == range_neuron.find_label_connection().value and x.value < min(range_neuron.value)])
                    if coeff > 0:
                        extensions.append([coeff, new_neuron, range_neuron])

        extensions = sorted(extensions, key=lambda x:x[0])
        extensions.reverse()
        if len(extensions) == 0:
            return False

        for extension in extensions:
            success = self.check_discrimination_after_adding(extension[2], extension[1]) > self.max_discrimination
            if success:
                return True

        return False

    def check_discrimination_after_adding(self, range_neuron, neuron_to_add):
        range_neuron.extend(neuron_to_add)
        all_correlated_out = list()
        for connected_range_neuron in self.get_range_connections():
            all_correlated_out += connected_range_neuron.get_out_objects_correlated()

        most_bad_correlations = 0
        if len(all_correlated_out) > 0:
            most_bad_correlations = all_correlated_out.count(mode(all_correlated_out))

        if len(self.get_range_connections()) - most_bad_correlations < self.max_discrimination:
            range_neuron.remove_connection(neuron_to_add)

        return len(self.get_range_connections()) - most_bad_correlations

    def is_represented(self, object):
        for represented in self.get_object_connections():
            if represented.value == object:
                return True

        return False

    def add_new_object_connections_if_represented(self):
        all_correlated_in = list()
        for range_neuron in self.get_range_connections():
            all_correlated_in.extend(range_neuron.get_in_objects_correlated(self.get_object_connections()))

        to_add = list()
        for object in set(all_correlated_in):
            if all_correlated_in.count(object) == len(self.get_range_connections()):
                to_add.append(object)

        for object in to_add:
            for connection in self.get_range_connections()[0].get_value_connections():
                for object_connection in connection.get_object_connections():
                    if object_connection.value == object:
                        self.add_connection_with_weight(1, object_connection)
                        object_connection.add_connection_with_weight(1, self)

    def get_represented_objects_quantity(self):
        return len([x for x in self.connections if x['neuron'].is_object()])

    def add_adef_weights(self, asonn):
        #7.38
        sum = 0
        for connection in self.connections:
            if connection['neuron'].is_range():
                a = connection['neuron'].calculate_7_34(asonn)
                sum += a

        for connection in self.connections:
            if connection['neuron'].is_range():
                b = connection['neuron'].calculate_7_34(asonn)
                weight = b / sum
                connection['weight'] = weight
                connection['neuron'].set_adef_weight(weight)

    def count_seeds(self):
        all_seeds = 0
        for connection in self.connections:
            if connection['neuron'].is_range():
                all_seeds += connection['neuron'].count_seeds()

        return all_seeds
