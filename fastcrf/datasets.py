class DatasetInfo(object):

    def __init__(self, dataset_description, name, class_name='y'):
        self.dataset_description = dataset_description
        self.inverse_dataset_description = {}
        self.name = name
        self.class_name = class_name

        for attr_pos, attr_info in self.dataset_description.items():
            if attr_info['type'] in (1, 2, 3):
                self.inverse_dataset_description[attr_info['original_position']] = attr_info
            else:
                for categ_idx, category_original_position in enumerate(attr_info['categories_original_position']):
                    self.inverse_dataset_description[category_original_position] = {
                        'type': 4,
                        'category_index': categ_idx,
                        'current_position': attr_info['current_position']
                    }

    def get_feature_type_features(self, feature_type):
        features = []
        for attr, attr_info in self.dataset_description.items():
            if attr_info['type'] == feature_type:
                features.append(attr_info['name'])
        return features


class DatasetInfoBuilder:

    def __init__(self, name):
        self.attributes = {}
        self.name = name

    def add_binary_variable(self, position, name='', category_names=()):
        self.attributes[position] = {'type': 3, 'name': name, 'range': 1, 'category_names': category_names,
                                     'lower_bound': 0, 'upper_bound': 1}

    def add_numerical_variable(self, position, lower_bound, upper_bound, name=''):
        variable_range = upper_bound - lower_bound
        self.attributes[position] = {'type': 1, 'range': variable_range, 'name': name, 'lower_bound': lower_bound,
                                     'upper_bound': upper_bound}

    def add_ordinal_variable(self, position, lower_bound, upper_bound, name=''):
        variable_range = upper_bound - lower_bound
        self.attributes[position] = {'type': 2, 'range': variable_range, 'name': name, 'lower_bound': lower_bound,
                                     'upper_bound': upper_bound}

    def add_one_hot_varible(self, start, num_categories, name='', category_names=()):
        for i in range(start, start + num_categories):
            self.attributes[i] = {'type': 4, 'main': start, 'name': name, 'category_names': category_names}

    def add_categorical_numerical(self, position, num_categories, name='', category_names=()):
        self.attributes[position] = {'type': 2, 'range': num_categories, 'name': name, 'category_names_mapping': category_names,
                                     'lower_bound': 0, 'upper_bound': num_categories, 'original_type': 5}

    def create_dataset_info(self):
        if set(self.attributes.keys()) != set(range(len(self.attributes))):
            raise ValueError(
                "Attributes cannot have unfilled positions. Unfilled {}".format(set(range(len(self.attributes)))
                                                                                - set(self.attributes.keys())))

        current_attr = 0
        dataset_description = {}
        previous_category = -1
        for attr_pos, description in self.attributes.items():
            if description['type'] in (1, 2, 3):
                if previous_category != -1:
                    current_attr += 1

                dataset_description[current_attr] = {'type': description['type'],
                                                     'range': description['range'],
                                                     'lower_bound': description['lower_bound'],
                                                     'upper_bound': description['upper_bound'],
                                                     'original_position': attr_pos,
                                                     'current_position': current_attr,
                                                     'name': description['name']}

                if 'category_names' in description:
                    dataset_description[current_attr]['category_names'] = description['category_names']

                current_attr += 1
                previous_category = -1
            else:
                if previous_category != -1 and previous_category != description['main']:
                    current_attr += 1

                if current_attr not in dataset_description:
                    dataset_description[current_attr] = {'type': 4,
                                                         'categories_original_position': [],
                                                         'current_position': current_attr,
                                                         'name': description['name'],
                                                         'category_names': description['category_names'],
                                                         'range': 0}

                dataset_description[current_attr]['categories_original_position'].append(attr_pos)
                previous_category = description['main']

        return DatasetInfo(dataset_description, self.name)
