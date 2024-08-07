from PySide6.QtCore import QObject, Signal


class Parameter(QObject):
    changed = Signal(object)

    def __init__(self, name=None, items=None, label=None, default=None, validator=None, groups=None, updater=None,
                 **kwargs):
        super().__init__()
        self.name = name
        self.items = {} if items is None else items
        self.label = label
        self._default = default
        self.validator = validator
        self.updater = updater
        self.updater_kwargs = kwargs
        self.groups = [] if groups is None else groups
        self._value = default

    @property
    def default(self):
        if callable(self._default):
            return self._default()
        return self._default

    @property
    def value(self):
        return self._value

    def set_value(self, value):
        if self.validate(value):
            self._value = value
            self.changed.emit(value)

    @value.setter
    def value(self, value):
        self.set_value(value)

    @property
    def item(self):
        item = self.get_item(self._value)
        if item is None:
            return self.value
        return item

    def reset(self):
        print(f'resetting {self.name} to {self.default}')
        self.value = self.default

    def update(self):
        if self.updater is not None:
            self.updater(**self.updater_kwargs)

    def set_items(self, items):
        if items is None:
            self.items = {}
        elif isinstance(items, list):
            self.items = {item: None for item in items}
        else:
            self.items = items

    def add_item(self, item):
        if not isinstance(item, dict):
            item = {item: None}
        self.items.update(item)

    def add_items(self, items):
        self.items.update(items)

    def get_item(self, key):
        if key in self.items:
            return self.items[key]
        return None

    def validate(self, value):
        """
        Abstract method that needs to be implemented in each concrete Parameter implementation to for validation
        of the new value
        """
        if self.items != {}:
            return value in self.items
        return (self.validator is None) or self.validator(value)


class Parameters:
    """
    A class to handle plugin parameters
    """

    def __init__(self, parameters=None):
        # self.plugin = plugin
        # self.param_groups = {'default': {}} if parameters is None else parameters
        if isinstance(parameters, list):
            self.parameter_list = parameters
        else:
            self.parameter_list = [parameters]

    def reset(self):
        for parameter in self.parameter_list:
            parameter.reset()

    def values(self, param_list=None):
        param_list = self.parameter_list if param_list is None else param_list
        return {param.name: param.value for param in param_list}

    def get_group(self, group):
        return [param for param in self.parameter_list if param.group == group]

    def get_value(self, name):
        return self.values()[name]

    def get(self, name):
        return self.to_dict()[name]

    def __repr__(self):
        return f'{self.values()}'

    def to_dict(self):
        return {param.name: param for param in self.parameter_list}

    # def add_groups(self, groups):
    #     """
    #     Add empty groups of parameters
    #
    #     :param groups: the list of groups
    #     """
    #     groups = groups if isinstance(groups, list) else [groups]
    #     for group in groups:
    #         self._add_group(group)
    #
    # def _add_group(self, group, param_dict=None):
    #     """
    #     Add a new group of parameters (in a dictionary)
    #
    #     :param group: the group to create
    #     :param param_dict: the dictionary of parameters
    #     """
    #     self.param_groups[group] = param_dict if param_dict is not None else {}
    #
    # def add(self, params, group='default'):
    #     """
    #     Add parameters (in a dictionary) to an existing group of parameters
    #
    #     :param group: the group of parameters to expand
    #     :param params: the parameters
    #     """
    #     params = params if isinstance(params, list) else [params]
    #
    #     if group in self.param_groups:
    #         self.param_groups[group].update({p.name: p for p in params})
    #     else:
    #         self._add_group(group, {p.name: p for p in params})
    #
    # def reset(self):
    #     for group in self.param_groups.keys():
    #         for param in self.param_groups[group].values():
    #             param.reset()
    #
    # def get_values(self, groups=None):
    #     """
    #     Get a dictionary containing all parameters key/values for a given group
    #
    #     :param group: the requested parameter group
    #     :return: a dictionary of parameters
    #     """
    #     if groups is None:
    #         groups = list(self.param_groups.keys())
    #         if groups == ['default']:
    #             return {name: param.value for name, param in self.param_groups['default'].items()}
    #     elif not isinstance(groups, list):
    #         groups = [groups]
    #     return {group: {name: param.value for name, param in self.param_groups[group].items()} for group in groups}
    #
    # def get_value(self, name, group='default'):
    #     return self.get_values(group)[group][name]
    #
    # def __repr__(self):
    #     return f'{self.get_values()}'
    #
    # def to_dict(self):
    #     groups = list(self.param_groups.keys())
    #     if groups == ['default']:
    #             return {name: param for name, param in self.param_groups['default'].items()}
    #     return {group: {name: param for name, param in self.param_groups[group].items()} for group in groups}
