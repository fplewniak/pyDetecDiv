import json

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
        try:
            return json.loads(self._value)
        except:
            return self._value

    def set_value(self, value):
        if value != self._value and self.validate(value):
            if isinstance(value, (list, dict)):
                value = json.dumps(value)
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
            # if isinstance(value, (list, dict)):
            #     return json.dumps(value) in self.items
            if not isinstance(value, (list, dict)):
                return value in self.items
        return (self.validator is None) or self.validator(value)


class Parameters:
    """
    A class to handle plugin parameters
    """

    def __init__(self, parameters=None):
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
