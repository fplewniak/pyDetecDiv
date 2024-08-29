import json

from PySide6.QtCore import QObject, Signal, QAbstractItemModel, QModelIndex, Qt

from pydetecdiv.app.models import ItemModel, DictItemModel


class Parameter:

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

    def clear(self):
        self.model.clear()

    @property
    def value(self):
        return self.model.value()

    def set_value(self, value):
        if value != self.model.value() and self.validate(value):
            if isinstance(value, (list, dict)):
                value = json.dumps(value)
            self.model.set_value(value)

    @value.setter
    def value(self, value):
        self.set_value(value)

    @property
    def json(self):
        return self.value

    def reset(self):
        self.value = self.default

    def update(self):
        if self.updater is not None:
            self.updater(**self.updater_kwargs)

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

    def validate(self, value):
        if self.validator is None:
            return self.minimum <= value <= self.maximum
        else:
            return self.validator(value)


class IntParameter(NumParameter):
    def __init__(self, name, label=None, default=None, minimum=1, maximum=4096, validator=None,
                 groups=None, updater=None, **kwargs):
        super().__init__(name=name, label=label, default=default, validator=validator, groups=groups, updater=updater,
                         minimum=minimum, maximum=maximum, **kwargs)

    def validate(self, value):
        if self.validator is None:
            return isinstance(value, int) & (self.minimum <= value <= self.maximum)
        else:
            return self.validator(value)


class FloatParameter(NumParameter):
    def __init__(self, name, label=None, default=None, validator=None, minimum=0.0, maximum=1.0,
                 groups=None, updater=None, **kwargs):
        super().__init__(name=name, label=label, default=default, validator=validator, groups=groups, updater=updater,
                         minimum=minimum, maximum=maximum, **kwargs)

    def validate(self, value):
        if self.validator is None:
            return isinstance(value, float) & (self.minimum <= value <= self.maximum)
        else:
            return self.validator(value)


class StringParameter(ItemParameter):
    def __init__(self, name, label=None, default='', validator=None,
                 groups=None, updater=None, **kwargs):
        super().__init__(name=name, label=label, default=default, validator=validator, groups=groups, updater=updater,
                         **kwargs)


class CheckParameter(ItemParameter):
    def __init__(self, name, label=None, default=False, validator=None, exclusive=True,
                 groups=None, updater=None, **kwargs):
        super().__init__(name=name, label=label, default=default, validator=validator, groups=groups, updater=updater,
                         **kwargs)
        self.exclusive = exclusive

    def kwargs(self):
        return {'default': self.default, 'exclusive': self.exclusive}


class ChoiceParameter(Parameter):
    def __init__(self, name, items=None, label=None, default=None, validator=None, groups=None, updater=None, **kwargs):
        super().__init__(name=name, label=label, default=default, validator=validator, groups=groups, updater=updater,
                         **kwargs)
        self.model = DictItemModel(items)

    @property
    def json(self):
        try:
            return json.loads(self.key)
        except json.decoder.JSONDecodeError:
            return self.key

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
        return self.model.rows()

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

    def reset(self, groups=None):
        for parameter in self.get_groups(groups):
            parameter.reset()
            # print(f'{parameter.name}: {parameter.value} ({parameter.default})')

    def update(self, groups=None):
        # for parameter in self.parameter_list:
        for parameter in self.get_groups(groups):
            parameter.update()

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

    def get_groups(self, groups, union=False):
        if groups is not None:
            if isinstance(groups, list):
                groups = set(groups)
            if isinstance(groups, str):
                groups = {groups}
            if union:
                return [param for param in self.parameter_list if param.groups.union(groups)]
            return [param for param in self.parameter_list if param.groups.intersection(groups)]
        return self.parameter_list

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
            return self_dict[item]
        raise KeyError

    def to_dict(self):
        return {param.name: param for param in self.parameter_list}

    def json(self, param_list=None, groups=None):
        if groups is None:
            if param_list is None:
                param_list = self.parameter_list
        else:
            group_params = self.get_groups(groups)
            if param_list is None:
                param_list = group_params
            else:
                param_list = list(set(param_list).intersection(set(group_params)))
        return {param.name: param.json for param in param_list}
