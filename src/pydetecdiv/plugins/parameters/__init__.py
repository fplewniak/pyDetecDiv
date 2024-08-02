from PySide6.QtCore import QObject, Signal


class Parameter(QObject):
    changed = Signal(object)

    def __init__(self, name=None, ptype=None, label=None, default=None, validator=None):
        super().__init__()
        self.name = name
        self.ptype = ptype
        self.label = label
        self.default = default
        self.validator = validator
        self._value = default

    @property
    def value(self):
        return self._value

    def set_value(self, value):
        self._value = value
        print(f'GUI => parameter {self._value}')

    @value.setter
    def value(self, value):
        if self.validator is None or self.validator(value):
            self._value = value
            self.changed.emit(value)

    def reset(self):
        self.value = self.default

    def validate(self, value):
        """
        Abstract method that needs to be implemented in each concrete Parameter implementation to for validation
        of the new value
        """
        raise NotImplementedError


class Parameters:
    """
    A class to handle plugin parameters
    """

    def __init__(self, plugin, parameters=None):
        self.plugin = plugin
        self.param_groups = {} if parameters is None else parameters

    def add_groups(self, groups):
        """
        Add empty groups of parameters

        :param groups: the list of groups
        """
        groups = groups if isinstance(groups, list) else [groups]
        for group in groups:
            self._add_group(group)

    def _add_group(self, group, param_dict=None):
        """
        Add a new group of parameters (in a dictionary)

        :param group: the group to create
        :param param_dict: the dictionary of parameters
        """
        self.param_groups[group] = param_dict if param_dict is not None else {}

    def add(self, group, params):
        """
        Add parameters (in a dictionary) to an existing group of parameters

        :param group: the group of parameters to expand
        :param params: the parameters
        """
        params = params if isinstance(params, list) else [params]

        if group in self.param_groups:
            self.param_groups[group].update({p.name: p for p in params})
        else:
            self._add_group(group, {p.name: p for p in params})

    def get_values(self, groups=None):
        """
        Get a dictionary containing all parameters key/values for a given group

        :param group: the requested parameter group
        :return: a dictionary of parameters
        """
        if groups is None:
            groups = self.param_groups.keys()
        elif not isinstance(groups, list):
            groups = [groups]
        return {group: {name: param.value for name, param in self.param_groups[group].items()} for group in groups}

    def get_value(self, name, group):
        return self.get_values(group)[group][name]

    def __repr__(self):
        return f'{self.get_values()}'
