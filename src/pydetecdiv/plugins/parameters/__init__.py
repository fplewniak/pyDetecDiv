# """
# Module defining the different types of parameters that may be needed to store information about a process and that may
# be specified using GUI widgets which are synchronized thanks to a shared model
# """
# import json
# from typing import Callable, Any
#
# from PySide6.QtCore import QAbstractItemModel
#
# from pydetecdiv.app.models import ItemModel, DictItemModel, StandardItemModel
#
#
# class Parameter:
#     """
#     Generic class defining the general behaviour of parameters
#     """
#
#     def __init__(self, name: str, label: str = None, default: Any = None, validator: Callable[[Any], bool] = None,
#                  groups: set[str] = None, updater: Callable = None, **kwargs: dict[str, Any]) -> None:
#         super().__init__()
#         self.name: str = name
#         self.label: str = label
#         self._default: Any = default
#         self.validator: Callable[[Any], bool] = validator
#         self.updater: Callable = updater
#         self.updater_kwargs: dict[str, Any] = kwargs
#         self.groups: set[str] = set() if groups is None else groups
#         self.model: StandardItemModel | None = None
#
#     def kwargs(self) -> dict[str, Any]:
#         """
#         Returns keywords arguments that should be passed to any widget used to manage the parameter. This method should
#         be overridden for specific Parameter implementations that may need more control over widgets
#
#         :return: a dictionary containing the keywords arguments
#         """
#         return {'default': self.default}
#
#     @property
#     def default(self) -> Any:
#         """
#         Returns the default value. If self._default is a callable, it is called to determine which value should be
#         returned
#
#         :return: the default value
#         """
#         if callable(self._default):
#             return self._default()
#         return self._default
#
#     def clear(self) -> None:
#         """
#         Clears the parameter of it's content
#         """
#         self.model.clear()
#
#     @property
#     def value(self) -> Any:
#         """
#         Returns the current value of the parameter
#
#         :return: the current value
#         """
#         return self.model.value()
#
#     def set_value(self, value: Any) -> None:
#         """
#         Sets the current value for the parameter. The model's value setter is called only if the new value is different
#         to avoid emit the changed signal without any reason
#
#         :param value: the new parameter value
#         """
#         if value != self.model.value() and self.validate(value):
#             if isinstance(value, (list, dict)):
#                 value = json.dumps(value)
#             self.value = value
#
#     @value.setter
#     def value(self, value: Any) -> None:
#         """
#         Setter for the current value of the parameter
#
#         :param value: the new parameter value
#         """
#         self.model.set_value(value)
#
#     @property
#     def json(self) -> Any:
#         """
#         Returns the current value in a json-compatible format. This method should be overridden for types that cannot
#         be dumped with json.dumps.
#
#         :return: the parameter value
#         """
#         return self.value
#
#     def reset(self) -> None:
#         """
#         Reset the value to the specified default
#         """
#         self.set_value(self.default)
#
#     def update(self) -> None:
#         """
#         Runs the specify updater callable if it was set, to update the parameter's value or choice
#         """
#         if self.updater is not None:
#             self.updater(**self.updater_kwargs)
#
#     def validate(self, value: Any) -> None:
#         """
#         Validates the value using the specified validator callable or always returning True if validator is None.
#         This method should be overridden for more specific needs of particular Parameter types
#         """
#         return (self.validator is None) or self.validator(value)
#
#
# class ItemParameter(Parameter):
#     """
#     Class representing a parameter holding any kind of single value parameter (item).
#     """
#
#     def __init__(self, name: str, label: str = None, default: Any = None, validator: Callable[..., bool] = None,
#                  groups: set[str] = None, updater: Callable = None, **kwargs: dict[str, Any]) -> None:
#         super().__init__(name=name, label=label, default=default, validator=validator, groups=groups, updater=updater,
#                          **kwargs)
#         self.model: ItemModel = ItemModel()
#         self.reset()
#
#
# class NumParameter(ItemParameter):
#     """
#     Class representing a parameter holding a number.
#     """
#
#     def __init__(self, name: str, label: str = None, default: int | float = None, minimum: int | float = None,
#                  maximum: int | float = None, validator: Callable[[int | float], bool] = None,
#                  groups: set[str] = None, updater: Callable = None, **kwargs: dict[str, Any]) -> None:
#         self.minimum: int | float = minimum
#         self.maximum: int | float = max(minimum, maximum)
#         super().__init__(name=name, label=label, default=default, validator=validator, groups=groups, updater=updater,
#                          **kwargs)
#
#     def kwargs(self) -> dict[str, Any]:
#         """
#         Returns keywords arguments that should be passed to any widget used to manage the parameter
#
#         :return: a dictionary containing the keywords arguments
#         """
#         return {'default': self.default, 'minimum': self.minimum, 'maximum': self.maximum}
#
#     def reset(self) -> None:
#         """
#         Resets the current value to its default it is specified
#         """
#         if self.default is None:
#             self.set_value(self.minimum)
#         else:
#             self.set_value(self.default)
#
#     def set_minimum(self, value: int | float) -> None:
#         """
#         Sets the minimum accepted value
#
#         :param value: the minimum value
#         """
#         self.minimum = value
#
#     def set_maximum(self, value: int | float) -> None:
#         """
#         Sets the maximum accepted value
#
#         :param value: the maximum value
#         """
#         self.maximum = value
#
#     def set_range(self, minimum: int | float, maximum: int | float) -> None:
#         """
#         Sets the range of accepted values for the numerical parameter
#
#         :param minimum: the minimum accepted value
#         :param maximum: the maximum accepted value
#         """
#         self.set_minimum(minimum)
#         self.set_maximum(maximum)
#
#     def validate(self, value: int | float) -> bool:
#         """
#         Validates a numerical value, returning True if value is numerical and lies within the specified range,
#         False otherwise
#
#         :param value: the value to be tested
#         :return: the result of the validation process
#         """
#         if self.validator is None:
#             return self.minimum <= value <= self.maximum
#         return self.validator(value)
#
#
# class IntParameter(NumParameter):
#     """
#     Class representing a parameter holding a integer number.
#     """
#
#     def __init__(self, name: str, label: str = None, default: int = 1, validator: Callable[[int], bool] = None,
#                  minimum: int = 1, maximum: int = 4096, groups: set[str] = None, updater: Callable = None,
#                  **kwargs: dict[str, Any]) -> None:
#         super().__init__(name=name, label=label, default=default, validator=validator, groups=groups, updater=updater,
#                          minimum=minimum, maximum=maximum, **kwargs)
#
#     def validate(self, value: int) -> bool:
#         """
#         Validates the value for a integer parameter, making sure it lies within the specified range
#
#         :param value: the value to be validated
#         :return: the result of the validation test: True if the value is validated, False otherwise
#         """
#         if self.validator is None:
#             return isinstance(value, int) & (self.minimum <= value <= self.maximum)
#         return self.validator(value)
#
#
# class FloatParameter(NumParameter):
#     """
#     Class representing a parameter holding a float number.
#     """
#
#     def __init__(self, name: str, label: str = None, default: float = 0.0, validator: Callable[[float], bool] = None,
#                  minimum: float = 0.0, maximum: float = 1.0, groups: set[str] = None, updater: Callable = None,
#                  **kwargs: dict[str, Any]) -> None:
#         super().__init__(name=name, label=label, default=default, validator=validator, groups=groups, updater=updater,
#                          minimum=minimum, maximum=maximum, **kwargs)
#
#     def validate(self, value: float) -> bool:
#         """
#         Validates the value for a float parameter, making sure it lies within the specified range
#
#         :param value: the value to be validated
#         :return: the result of the validation test: True if the value is validated, False otherwise
#         """
#         if self.validator is None:
#             return isinstance(value, float) & (self.minimum <= value <= self.maximum)
#         return self.validator(value)
#
#
# class StringParameter(ItemParameter):
#     """
#     Class representing a parameter holding text.
#     """
#
#     def __init__(self, name: str, label: str = None, default: str = '', validator: Callable[[str], bool] = None,
#                  groups: set[str] = None, updater: Callable = None, **kwargs: dict[str, Any]) -> None:
#         super().__init__(name=name, label=label, default=default, validator=validator, groups=groups, updater=updater,
#                          **kwargs)
#
#
# class CheckParameter(ItemParameter):
#     """
#     Class representing a parameter whose value is either True (checked) or False (unchecked). Such parameters can be
#     linked to CheckBox arranged in the same GroupBox and set to be mutually exclusive.
#     """
#
#     def __init__(self, name: str, label: str = None, exclusive: bool = True,
#                  default: str | int | float | bool | Callable = None, validator: Callable[[bool], bool] = None,
#                  groups: set[str] = None, updater: Callable[..., None] = None, **kwargs: dict[str, Any]) -> None:
#         super().__init__(name=name, label=label, default=default, validator=validator, groups=groups, updater=updater,
#                          **kwargs)
#         self.exclusive: bool = exclusive
#
#     def kwargs(self) -> dict[str, Any]:
#         """
#         Returns keywords arguments that should be passed to any widget used to manage the parameter
#
#         :return: a dictionary containing the keyword arguments
#         """
#         return {'default': self.default, 'exclusive': self.exclusive}
#
#
# class ChoiceParameter(Parameter):
#     """
#     Class representing a parameter whose value is a selection among several options (items)
#     """
#
#     def __init__(self, name: str, items: dict[str, object] = None, label: str = None,
#                  default: str | int | float | bool | Callable = None, validator: Callable[[Any], bool] = None,
#                  groups: set[str] = None, updater: Callable[..., None] = None, **kwargs: dict[str, Any]) -> None:
#         super().__init__(name=name, label=label, default=default, validator=validator, groups=groups, updater=updater,
#                          **kwargs)
#         self.model: DictItemModel = DictItemModel(items)
#
#     @property
#     def json(self) -> str | list | dict:
#         """
#         Returns the selected key in a way that is compatible with json.dumps. This is important to maintain consistency
#         in the way keys strings representing lists or dictionaries are stored in json format (in SQLite, etc).
#
#         :return: the json-compatible representation of selected key
#         """
#         try:
#             return json.loads(self.key)
#         except json.decoder.JSONDecodeError:
#             return self.key
#
#     @property
#     def key(self) -> str:
#         """
#         Returns the key of the selected choice
#
#         :return: the selected key
#         """
#         return self.model.key()
#
#     @property
#     def value(self) -> Any:
#         """
#         Returns the value object of the selected choice
#
#         :return: the selected value object
#         """
#         return self.model.value()
#
#     @value.setter
#     def value(self, key: str) -> None:
#         """
#         Setter for the selected choice by specifying the corresponding key
#
#         :param key: the key of the item to select
#         """
#         self.model.set_value(key)
#
#     @property
#     def keys(self) -> list[str]:
#         """
#         Returns all the keys for the ChoiceParameter corresponding to the actual choice values
#
#         :return: the list of all possible choices
#         """
#         return self.model.keys()
#
#     @property
#     def values(self) -> list[object]:
#         """
#         Returns all the choice values for the ChoiceParameter
#
#         :return: the list of all possible values
#         """
#         return self.model.values()
#
#     @property
#     def items(self) -> dict[str, object]:
#         """
#         Returns all choice items for this parameter as a dictionary with key = name/representation of the corresponding
#         option, value = the actual object
#
#         :return: all choice items
#         """
#         return self.model.rows()
#
#     @property
#     def item(self) -> object:
#         """
#         Return the object represented by the ChoiceParameter option
#
#         :return: the object represented by this option
#         """
#         return self.model.value()
#
#     def set_value(self, value: Any) -> None:
#         """
#         Sets the current value for the parameter. The model's value setter is called only if the new value is different
#         to avoid emit the changed signal without any reason
#
#         :param value: the new parameter selected key, pointing to the new value
#         """
#         if value is None:
#             value = self.keys[0]
#         if value != self.model.key() and self.validate(value):
#             if isinstance(value, (list, dict)):
#                 value = json.dumps(value)
#             self.model.set_value(value)
#
#     def set_items(self, items: dict[str, object]) -> None:
#         """
#         Sets the choice items of the ChoiceParameter object. It the object contained items before, these are cleared and
#         replaced with the new set.
#
#         :param items: a dictionary containing the items to add to the ChoiceParameter. Key is the name/representation
#          of the corresponding option, value is the actual object
#         """
#         self.model.set_items(items)
#
#     def add_item(self, item: dict[str, object]) -> None:
#         """
#         Add an item to the ChoiceParameter object
#
#         :param item: the dictionary containing the choice item to add to the ChoiceParameter. Key is the
#          name/representation of the corresponding option, value is the actual object.
#         """
#         self.model.add_item(item)
#
#     def add_items(self, items: dict[str, object]) -> None:
#         """
#         Add items to the ChoiceParameter object
#
#         :param items: a dictionary containing the choice items to add to the ChoiceParameter. Key is the
#          name/representation of the corresponding option, value is the actual object.
#         """
#         for k, v in items.items():
#             self.add_item({k: v})
#
#
# class Parameters:
#     """
#     A class to handle a list of parameters
#     """
#
#     def __init__(self, parameters: list[Parameter] | Parameter = None) -> None:
#         if isinstance(parameters, list):
#             self.parameter_list: list = parameters
#         else:
#             self.parameter_list: list = [parameters]
#
#     def reset(self, groups: list[str] | str = None) -> None:
#         """
#         Reset all parameters in groups
#
#         :param groups: the groups to reset
#         """
#         for parameter in self.get_groups(groups):
#             parameter.reset()
#
#     def update(self, groups: list[str] | str = None) -> None:
#         """
#         Updates all parameters in groups
#
#         :param groups: the groups to update
#         """
#         for parameter in self.get_groups(groups):
#             parameter.update()
#
#     def values(self, param_list: list[Parameter] = None, groups: list[str] | str = None) -> dict[str, Parameter]:
#         """
#         Returns a dictionary of parameters with name as key and value as value
#
#         :param param_list: the list of parameters, if None, all parameters are returned
#         :param groups: the parameters groups, if None, all parameters are returned
#         :return: a dictionary of all parameters
#         """
#         if groups is None:
#             param_list = param_list or self.parameter_list
#         else:
#             group_params = self.get_groups(groups)
#             param_list = [param for param in (param_list or group_params) if param in group_params]
#         return {param.name: param.value for param in param_list}
#
#     def get_groups(self, groups: list[str] | str, union: bool = False) -> list[Parameter]:
#         """
#         Get parameters from groups. If multiple groups are given, either intersection (default) or union of the
#         parameters sets is returned
#
#         :param groups: the groups
#         :param union: a boolean, if True, a parameter is returned if it belongs to at least one group, if False, it is
#          returned only if it belongs to all groups
#
#         :return: a list of parameters
#         """
#         if groups is None:
#             return self.parameter_list
#         if isinstance(groups, str):
#             groups = {groups}
#         if union:
#             return [param for param in self.parameter_list if param.groups.union(groups)]
#         return [param for param in self.parameter_list if param.groups.intersection(groups)]
#
#     def __repr__(self) -> str:
#         """
#         Return the parameters as a string
#
#         :rtype: str
#         """
#         return f'{self.values()}'
#
#     def __getitem__(self, item: str) -> Parameter:
#         """
#         Private method enabling Parameters to behave as if it were a dictionary of Parameter objects indexed by their
#         name
#
#         :rtype: Parameter
#         """
#         self_dict = self.to_dict()
#         if not isinstance(item, str):
#             raise TypeError
#         if item in self_dict:
#             return self_dict[item]
#         raise KeyError
#
#     def to_dict(self) -> dict[str, Parameter]:
#         """
#         Return the parameters as a dictionary
#
#         :return: a dictionary of all parameters, values are Parameter objects
#         """
#         return {param.name: param for param in self.parameter_list}
#
#     def json(self, param_list: list[Parameter] = None, groups: list[str] | str = None) -> dict[str, object]:
#         """
#         Return dictionary of parameters representing each of them as a json-compatible object
#
#         :param param_list: the list of parameters, if None, all parameters are returned
#         :param groups: the parameters groups, if None, all parameters are returned
#         :return: a dictionary of all parameters in a json-compatible format
#         """
#         if groups is None:
#             if param_list is None:
#                 param_list = self.parameter_list
#         else:
#             group_params = self.get_groups(groups)
#             if param_list is None:
#                 param_list = group_params
#             else:
#                 param_list = list(set(param_list).intersection(set(group_params)))
#         return {param.name: param.json for param in param_list}
