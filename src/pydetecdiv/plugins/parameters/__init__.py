import json

from PySide6.QtCore import QObject, Signal, QAbstractItemModel, QModelIndex, Qt

from pydetecdiv.app.models import ItemModel, DictItemModel


class Parameter:
    # changed = Signal(object)
    # itemsChanged = Signal(object)

    def __init__(self, name, label=None, default=None, validator=None, groups=None, updater=None, **kwargs):
        super().__init__()
        self.name = name
        self.label = label
        self._default = default
        self.validator = validator
        self.updater = updater
        self.updater_kwargs = kwargs
        self.groups = set() if groups is None else groups
        self.model = None

    def kwargs(self):
        return {'default': self.default}

    @property
    def default(self):
        if callable(self._default):
            return self._default()
        return self._default

    @property
    def value(self):
        return self.model.value()
        # try:
        #     return json.loads(self._value)
        # except:
        #     return self._value

    def set_value(self, value):
        if value != self.model.value() and self.validate(value):
            if isinstance(value, (list, dict)):
                value = json.dumps(value)
            self.model.set_value(value)

    @value.setter
    def value(self, value):
        self.set_value(value)

    # @property
    # def values(self):
    #     return list(self.items.keys())

    # @property
    # def data_list(self):
    #     return list(self.items.values())

    # @property
    # def item(self):
    #     item = self.get_item(self._value)
    #     if item is None:
    #         return self.value
    #     return item

    def reset(self):
        self.value = self.default

    def update(self):
        if self.updater is not None:
            self.updater(**self.updater_kwargs)

    # def set_items(self, items):
    #     if items is None:
    #         self.items = {}
    #     elif isinstance(items, list):
    #         self.items = {item: None for item in items}
    #     else:
    #         self.items = items
    #     # self.itemsChanged.emit(self.items)
    #
    # def add_item(self, item):
    #     if not isinstance(item, dict):
    #         item = {item: None}
    #     self.items.update(item)
    #     # self.itemsChanged.emit(self.items)
    #
    # def add_items(self, items):
    #     self.items.update(items)
    #     # self.itemsChanged.emit(self.items)
    #
    # def get_item(self, key):
    #     if key in self.items:
    #         return self.items[key]
    #     return None

    def validate(self, value):
        """
        Abstract method that needs to be implemented in each concrete Parameter implementation to for validation
        of the new value
        """
        return (self.validator is None) or self.validator(value)


class ItemParameter(Parameter):
    def __init__(self, name, label=None, default=None, validator=None, groups=None, updater=None, **kwargs):
        super().__init__(name=name, label=label, default=default, validator=validator, groups=groups, updater=updater,
                         **kwargs)
        self.model = ItemModel()
        self.reset()


class NumParameter(ItemParameter):
    def __init__(self, name, label=None, default=None, minimum=None, maximum=None, validator=None,
                 groups=None, updater=None, **kwargs):
        self.minimum = minimum
        self.maximum = max(minimum, maximum)
        super().__init__(name=name, label=label, default=default, validator=validator, groups=groups, updater=updater,
                         **kwargs)

    def kwargs(self):
        return {'default': self.default, 'minimum': self.minimum, 'maximum': self.maximum}

    def reset(self):
        if self.default is None:
            self.set_value(self.minimum)
        else:
            self.set_value(self.default)

    def set_minimum(self, value):
        self.minimum = value

    def set_maximum(self, value):
        self.maximum = value

    def set_range(self, minimum, maximum):
        self.set_minimum(minimum)
        self.set_maximum(maximum)


class IntParameter(NumParameter):
    def __init__(self, name, label=None, default=None, minimum=1, maximum=4096, validator=None,
                 groups=None, updater=None, **kwargs):
        super().__init__(name=name, label=label, default=default, validator=validator, groups=groups, updater=updater,
                         minimum=minimum, maximum=maximum, **kwargs)


class FloatParameter(NumParameter):
    def __init__(self, name, label=None, default=None, validator=None, minimum=0.0, maximum=1.0,
                 groups=None, updater=None, **kwargs):
        super().__init__(name=name, label=label, default=default, validator=validator, groups=groups, updater=updater,
                         minimum=minimum, maximum=maximum, **kwargs)


class StringParameter(ItemParameter):
    def __init__(self, name, label=None, default='', validator=None,
                 groups=None, updater=None, **kwargs):
        super().__init__(name=name, label=label, default=default, validator=validator, groups=groups, updater=updater,
                         **kwargs)
        self.model = ItemModel()
        self.reset()


class CheckParameter(ItemParameter):
    def __init__(self, name, label=None, default=None, validator=None,
                 groups=None, updater=None, **kwargs):
        super().__init__(name=name, label=label, default=default, validator=validator, groups=groups, updater=updater,
                         **kwargs)


class ChoiceParameter(Parameter):
    def __init__(self, name, items=None, label=None, default=None, validator=None, groups=None, updater=None, **kwargs):
        super().__init__(name=name, label=label, default=default, validator=validator, groups=groups, updater=updater,
                         **kwargs)
        self.model = DictItemModel(items)

    @property
    def key(self):
        return self.model.key()

    @property
    def value(self):
        return self.model.value()

    @value.setter
    def value(self, value):
        self.model.set_value(value)

    @property
    def keys(self):
        return self.model.keys()

    @property
    def values(self):
        return self.model.values()

    @property
    def items(self):
        return self.model.values()

    @property
    def item(self):
        return self.model.value()

    def set_items(self, items):
        self.model.set_items(items)

    def add_item(self, item):
        self.model.add_item(item)

    def add_items(self, items):
        for item in items.items():
            self.add_item(item)


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

    def values(self, param_list=None, groups=None):
        if groups is None:
            if param_list is None:
                param_list = self.parameter_list
        else:
            group_params = self.get_groups(groups)
            if param_list is None:
                param_list = group_params
            else:
                param_list = list(set(param_list).intersection(set(group_params)))
        return {param.name: param.value for param in param_list}

    def get_groups(self, groups):
        if isinstance(groups, list):
            groups = set(groups)
        if isinstance(groups, str):
            groups = {groups}
        return [param for param in self.parameter_list if param.groups.intersection(groups)]

    def get_value(self, name):
        return self.values()[name]

    def get(self, name):
        return self.to_dict()[name]

    def __repr__(self):
        return f'{self.values()}'

    def __getitem__(self, item):
        self_dict = self.to_dict()
        if not isinstance(item, str):
            raise TypeError
        elif item in self_dict:
            return self.to_dict()[item]
        raise KeyError

    def to_dict(self):
        return {param.name: param for param in self.parameter_list}
