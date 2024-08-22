import json

from PySide6.QtCore import QObject, Signal, QAbstractItemModel, QModelIndex, Qt


class Parameter(QAbstractItemModel):
    changed = Signal(object)
    itemsChanged = Signal(object)

    def __init__(self, name=None, items=None, label=None, default=None, validator=None, groups=None, updater=None,
                 **kwargs):
        super().__init__()
        self.name = name
        # self.items = {} if items is None else items
        self.set_items(items)
        self.label = label
        self._default = default
        self.validator = validator
        self.updater = updater
        self.updater_kwargs = kwargs
        self.groups = set() if groups is None else groups
        self._value = default

    def rowCount(self, parent=QModelIndex()):
        return 1  # Une seule ligne

    def columnCount(self, parent=QModelIndex()):
        return 1  # Une seule colonne

    def data(self, index, role=Qt.DisplayRole):
        if role == Qt.DisplayRole or role == Qt.EditRole:
            return self._value
        return None

    def setData(self, index, value, role=Qt.EditRole):
        if role == Qt.EditRole:
            self._value = value
            self.dataChanged.emit(index, index, [role])
            return True
        return False

    def index(self, row, column, parent=QModelIndex()):
        if not parent.isValid() and row == 0 and column == 0:
            return self.createIndex(row, column)
        return QModelIndex()

    def parent(self, index):
        return QModelIndex()

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
            # self._value = value
            self.setData(self.index(0,0), value)
            self.changed.emit(value)

    @value.setter
    def value(self, value):
        self.set_value(value)

    @property
    def values(self):
        return list(self.items.keys())

    @property
    def data_list(self):
        return list(self.items.values())

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
        self.itemsChanged.emit(self.items)

    def add_item(self, item):
        if not isinstance(item, dict):
            item = {item: None}
        self.items.update(item)
        self.itemsChanged.emit(self.items)

    def add_items(self, items):
        self.items.update(items)
        self.itemsChanged.emit(self.items)

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
