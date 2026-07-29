"""
Module defining the different types of parameters that may be needed to store information about a process and that may
be specified using GUI widgets which are synchronized thanks to a shared model
"""
import json
from typing import Callable, Any, cast, TypeVar, Generic

from PySide6.QtCore import SignalInstance

from pydetecdiv.app.models import ItemModel, DictItemModel, StandardItemModel, StringListModel

Num = TypeVar('Num', float, int)

class Parameter:
    """
    Generic class defining the general behaviour of parameters
    """

    def __init__(self, name: str, label: str | None = None, default: Any = None, validator: Callable[[Any], bool] | None = None,
                 groups: set[str] | None = None, updater: Callable | None = None, commands: set[str] | None = None,
                 **kwargs: dict[str, Any]) -> None:
        super().__init__()
        self.name: str = name
        self.label: str | None = label
        self._default: Any = default
        self.validator: Callable[[Any], bool] | None = validator
        self.updater: Callable | None = updater
        self.updater_kwargs: dict[str, Any] = kwargs
        self.groups: set[str] = set() if groups is None else groups
        self.qmodel: StandardItemModel = StandardItemModel()
        # self.should_be_saved: bool = False
        self.commands: set[str] = set() if commands is None else commands
        self.__dict__.update(kwargs)

    def kwargs(self) -> dict[str, Any]:
        """
        Returns keywords arguments that should be passed to any widget used to manage the parameter. This method should
        be overridden for specific Parameter implementations that may need more control over widgets

        :return: a dictionary containing the keywords arguments
        """
        return self.__dict__
        # return {'default': self.default}

    @property
    def default(self) -> Any:
        """
        Returns the default value. If self._default is a callable, it is called to determine which value should be
        returned

        :return: the default value
        """
        if callable(self._default):
            return self._default()
        return self._default

    def clear(self) -> None:
        """
        Clears the parameter of it's content
        """
        if self.qmodel is not None:
            self.qmodel.clear()

    @property
    def value(self) -> Any:
        """
        Returns the current value of the parameter

        :return: the current value
        """
        if self.qmodel is not None:
            return self.qmodel.value()
        return None

    @value.setter
    def value(self, value: Any) -> None:
        """
        Setter for the current value of the parameter

        :param value: the new parameter value
        """
        if self.qmodel is not None:
            self.qmodel.set_value(value)

    @property
    def type(self) -> str:
        """
        Return the parameter's type

        :return: the parameter's type
        """
        return self.__class__.__name__

    def set_value(self, value: Any) -> None:
        """
        Sets the current value for the parameter. The model's value setter is called only if the new value is different
        to avoid emit the changed signal without any reason

        :param value: the new parameter value
        """
        if value != self.qmodel.value() and self.validate(value):
            if isinstance(value, (list, dict)):
                value = json.dumps(value)
            self.value = value

    @property
    def json(self) -> Any:
        """
        Returns the current value in a json-compatible format. This method should be overridden for types that cannot
        be dumped with json.dumps.

        :return: the parameter value
        """
        return self.value

    def reset(self) -> None:
        """
        Reset the value to the specified default
        """
        self.set_value(self.default)

    def update(self) -> None:
        """
        Runs the specify updater callable if it was set, to update the parameter's value or choice
        """
        if self.updater is not None:
            self.updater(**self.updater_kwargs)

    def validate(self, value: Any) -> bool:
        """
        Validates the value using the specified validator callable or always returning True if validator is None.
        This method should be overridden for more specific needs of particular Parameter types
        """
        return (self.validator is None) or self.validator(value)

    @property
    def changed(self) -> SignalInstance:
        """
        return property telling whether the spinbox value has changed. This overwrites the Pyside equivalent method in
         order to have the same method name for all widgets

        :return: boolean indication whether the value has changed
        """
        return self.qmodel.itemChanged

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Parameter):
            return self.value == other.value
        return self.value == other

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other: Any) -> bool:
        if isinstance(other, Parameter):
            return self.value < other.value
        return self.value < other

    def __le__(self, other: Any) -> bool:
        if isinstance(other, Parameter):
            return self.value <= other.value
        return self.value <= other

    def __gt__(self, other: Any) -> bool:
        return not self.__le__(other)

    def __ge__(self, other: Any) -> bool:
        return not self.__lt__(other)

    def __add__(self, other: Any) -> Any:
        if isinstance(other, Parameter):
            return self.value + other.value
        return self.value + other

    def __radd__(self, other: Any) -> Any:
        return self.__add__(other)

    def __sub__(self, other: Any) -> Any:
        if isinstance(other, Parameter):
            return self.value - other.value
        return self.value - other

    def __rsub__(self, other: Any) -> Any:
        return self.__sub__(other)

    def __mul__(self, other: Any) -> Any:
        if isinstance(other, Parameter):
            return self.value * other.value
        return self.value * other

    def __rmul__(self, other: Any) -> Any:
        return self.__mul__(other)

    def __truediv__(self, other: Any) -> Any:
        if isinstance(other, Parameter):
            return self.value / other.value
        return self.value / other

    def __rtruediv__(self, other: Any) -> Any:
        return self.__truediv__(other)

    def __bool__(self) -> bool:
        return bool(self.value)

    def __hash__(self) -> int:
        return hash(self.name)

    def __repr__(self) -> str:
        return str(self.value)


class ItemParameter(Parameter):
    """
    Class representing a parameter holding any kind of single value parameter (item).
    """

    def __init__(self, name: str, label: str | None = None, default: Any | None = None,
                 validator: Callable[..., bool] | None = None, groups: set[str] | None = None, updater: Callable | None = None,
                 commands: set[str] | None = None, **kwargs: dict[str, Any]) -> None:
        super().__init__(name=name, label=label, default=default, validator=validator, groups=groups, updater=updater,
                         commands=commands, **kwargs)
        self.qmodel: ItemModel = ItemModel()
        # self.reset()


class NumParameter(ItemParameter, Generic[Num]):
    """
    Class representing a parameter holding a number.
    """
    def __init__(self, name: str, label: str | None = None, default: int | float | None = None, minimum: int | float | None = None,
                 maximum: int | float | None = None, validator: Callable[[Num], bool] | None = None,
                 groups: set[str] | None = None, updater: Callable | None = None, commands: set[str] | None = None,
                 **kwargs: dict[str, Any]) -> None:
        self.minimum: int | float | None = minimum
        self.maximum: int | float | None = max(cast(int | float, minimum), cast(int | float, maximum))
        super().__init__(name=name, label=label, default=default, validator=validator, groups=groups, updater=updater,
                         commands=commands, **kwargs)

    # def kwargs(self) -> dict[str, Any]:
    #     """
    #     Returns keywords arguments that should be passed to any widget used to manage the parameter
    #
    #     :return: a dictionary containing the keywords arguments
    #     """
    #     return {'default': self.default, 'minimum': self.minimum, 'maximum': self.maximum}

    def reset(self) -> None:
        """
        Resets the current value to its default it is specified
        """
        if self.default is None:
            self.set_value(self.minimum)
        else:
            self.set_value(self.default)

    def set_minimum(self, value: int | float) -> None:
        """
        Sets the minimum accepted value

        :param value: the minimum value
        """
        self.minimum = value

    def set_maximum(self, value: int | float) -> None:
        """
        Sets the maximum accepted value

        :param value: the maximum value
        """
        self.maximum = value

    def set_range(self, minimum: int | float, maximum: int | float) -> None:
        """
        Sets the range of accepted values for the numerical parameter

        :param minimum: the minimum accepted value
        :param maximum: the maximum accepted value
        """
        self.set_minimum(minimum)
        self.set_maximum(maximum)

    def validate(self, value: Num) -> bool:
        """
        Validates a numerical value, returning True if value is numerical and lies within the specified range,
        False otherwise

        :param value: the value to be tested
        :return: the result of the validation process
        """
        if self.validator is None:
            return self.minimum <= value <= self.maximum
        return self.validator(value)


class IntParameter(NumParameter[int]):
    """
    Class representing a parameter holding a integer number.
    """

    def __init__(self, name: str, label: str | None = None, default: int = 1, validator: Callable[[int], bool] | None = None,
                 minimum: int = 1, maximum: int = 4096, groups: set[str] | None = None, updater: Callable | None = None,
                 commands: set[str] | None = None, **kwargs: dict[str, Any]) -> None:
        super().__init__(name=name, label=label, default=default, validator=validator, groups=groups, updater=updater,
                         minimum=minimum, maximum=maximum, commands=commands, **kwargs)

    # def kwargs(self) -> dict[str, Any]:
    #     """
    #     Returns keywords arguments that should be passed to any widget used to manage the parameter. This method should
    #     be overridden for specific Parameter implementations that may need more control over widgets
    #
    #     :return: a dictionary containing the keywords arguments
    #     """
    #     return {'default': self.default, 'minimum': self.minimum, 'maximum': self.maximum, 'single_step': self.single_step}


    def validate(self, value: int) -> bool:
        """
        Validates the value for a integer parameter, making sure it lies within the specified range

        :param value: the value to be validated
        :return: the result of the validation test: True if the value is validated, False otherwise
        """
        if self.validator is None:
            return isinstance(value, int) & (cast(int, self.minimum) <= value <= cast(int, self.maximum))
        return self.validator(value)


class FloatParameter(NumParameter[float]):
    """
    Class representing a parameter holding a float number.
    """

    def __init__(self, name: str, label: str | None = None, default: float = 0.0,
                 validator: Callable[[float], bool] | None = None, minimum: float = 0.0, maximum: float = 1.0,
                 groups: set[str] | None = None, updater: Callable | None = None, commands: set[str] | None = None,
                 **kwargs: dict[str, Any]) -> None:
        super().__init__(name=name, label=label, default=default, validator=validator, groups=groups, updater=updater,
                         minimum=minimum, maximum=maximum, commands=commands, **kwargs)

    def validate(self, value: float) -> bool:
        """
        Validates the value for a float parameter, making sure it lies within the specified range

        :param value: the value to be validated
        :return: the result of the validation test: True if the value is validated, False otherwise
        """
        if self.validator is None:
            return isinstance(value, float) & (self.minimum <= value <= self.maximum)
        return self.validator(value)


class StringParameter(ItemParameter):
    """
    Class representing a parameter holding text.
    """

    def __init__(self, name: str, label: str | None = None, default: str = '', validator: Callable[[str], bool] | None = None,
                 groups: set[str] | None = None, updater: Callable | None = None, commands: set[str] | None = None,
                 **kwargs: dict[str, Any]) -> None:
        super().__init__(name=name, label=label, default=default, validator=validator, groups=groups, updater=updater,
                         commands=commands, **kwargs)


class StringListParameter(Parameter):
    """
    A parameter defining a list of strings
    """
    def __init__(self, name: str, label: str | None = None, default: list[str] | None = None,
                 validator: Callable[[str], bool] | None = None, groups: set[str] | None = None, updater: Callable | None = None,
                 commands: set[str] | None = None, **kwargs: dict[str, Any]) -> None:
        super().__init__(name=name, label=label, default=default, validator=validator, groups=groups, updater=updater,
                         commands=commands, **kwargs)

        self.qmodel: StringListModel = StringListModel()
        # self.reset()

    def set_value(self, value: list[str]) -> None:
        self.value = value

    def append(self, item: str) -> None:
        """
        Append a string to the list

        :param item: the string item to add
        """
        self.qmodel.add_item(item)


class FileParameter(ItemParameter):
    """
    A parameter for choosing a file path.
    """
    def __init__(self, name: str, label: str | None = None, default: str | Callable = '',
                 validator: Callable[[str], bool] | None = None, groups: set[str]  | None = None, updater: Callable | None = None,
                 commands: set[str]  | None = None, require_existing: bool = False, **kwargs) -> None:
        super().__init__(name=name, label=label, default=default, validator=validator, groups=groups, updater=updater,
                         commands=commands, **kwargs)
        self.require_existing = require_existing


class DirParameter(ItemParameter):
    """
    A parameter for choosing a directory path.
    """
    def __init__(self, name: str, label: str | None = None, default: str | Callable = '.',
                 validator: Callable[[str], bool] | None = None, groups: set[str]  | None = None, updater: Callable | None = None,
                 commands: set[str]  | None = None, **kwargs) -> None:
        super().__init__(name=name, label=label, default=default, validator=validator, groups=groups, updater=updater,
                         commands=commands, **kwargs)

# class PathParameter(ItemParameter):
#     """
#     Class representing a parameter holding a path.
#     """
#     def __init__(self, name: str, label: str | None = None, default: str = '', validator: Callable[[str], bool] | None = None,
#                  groups: set[str] | None = None, updater: Callable | None = None, current_dir: str = '.',
#                  filters = list[str] | None, select_dir: bool = False, commands: set[str] | None = None,
#                  **kwargs) -> None:
#         super().__init__(name=name, label=label, default=default, validator=validator, groups=groups, updater=updater,
#                          commands=commands, **kwargs)
#         self.select_dir = select_dir
#         self.current_dir = current_dir
#         # self.filters = filters
#
#     def reset(self):
#         self.qmodel.set_value(os.path.join(self.current_dir, self.default))

class CheckParameter(ItemParameter):
    """
    Class representing a parameter whose value is either True (checked) or False (unchecked). Such parameters can be
    linked to CheckBox arranged in the same GroupBox and set to be mutually exclusive.
    """

    def __init__(self, name: str, label: str | None = None, exclusive: bool = True,
                 default: str | int | float | bool | Callable | None = None, validator: Callable[[bool], bool] | None = None,
                 groups: set[str] | None = None, updater: Callable[..., None] | None = None, commands: set[str] | None = None,
                 **kwargs: dict[str, Any]) -> None:
        super().__init__(name=name, label=label, default=default, validator=validator, groups=groups, updater=updater,
                         commands=commands, **kwargs)
        self.exclusive: bool = exclusive

    # def kwargs(self) -> dict[str, Any]:
    #     """
    #     Returns keywords arguments that should be passed to any widget used to manage the parameter
    #
    #     :return: a dictionary containing the keyword arguments
    #     """
    #     return {'default': self.default, 'exclusive': self.exclusive}


class ChoiceParameter(Parameter):
    """
    Class representing a parameter whose value is a selection among several options (items)
    """

    def __init__(self, name: str, items: dict[str, object] | None = None, label: str | None = None,
                 default: str | int | float | bool | Callable | None = None, validator: Callable[[Any], bool] | None = None,
                 groups: set[str] | None = None, updater: Callable[..., None] | None = None, commands: set[str] | None= None,
                 **kwargs: dict[str, Any]) -> None:
        super().__init__(name=name, label=label, default=default, validator=validator, groups=groups, updater=updater,
                         commands=commands, **kwargs)
        self.qmodel: DictItemModel = DictItemModel(items)

    # def kwargs(self) -> dict[str, Any]:
    #     """
    #     Returns keywords arguments that should be passed to any widget used to manage the parameter
    #
    #     :return: a dictionary containing the keyword arguments
    #     """
    #     return {'default': self.default}

    @property
    def json(self) -> str | list | dict:
        """
        Returns the selected key in a way that is compatible with json.dumps. This is important to maintain consistency
        in the way keys strings representing lists or dictionaries are stored in json format (in SQLite, etc).

        :return: the json-compatible representation of selected key
        """
        try:
            return json.loads(cast(str, self.key))
        except json.decoder.JSONDecodeError:
            return cast(str, self.key)

    @property
    def key(self) -> str | None :
        """
        Returns the key of the selected choice

        :return: the selected key
        """
        return self.qmodel.key()

    @property
    def value(self) -> Any:
        """
        Returns the value object of the selected choice

        :return: the selected value object
        """
        return self.qmodel.value()

    @value.setter
    def value(self, key: str) -> None:
        """
        Setter for the selected choice by specifying the corresponding key

        :param key: the key of the item to select
        """
        self.qmodel.set_value(key)

    @property
    def keys(self) -> list[str]:
        """
        Returns all the keys for the ChoiceParameter corresponding to the actual choice values

        :return: the list of all possible choices
        """
        return self.qmodel.keys()

    @property
    def values(self) -> list[object]:
        """
        Returns all the choice values for the ChoiceParameter

        :return: the list of all possible values
        """
        return self.qmodel.values()

    @property
    def items(self) -> dict[str, Any]:
        """
        Returns all choice items for this parameter as a dictionary with key = name/representation of the corresponding
        option, value = the actual object

        :return: all choice items
        """
        return self.qmodel.rows().items()

    @property
    def item(self) -> object:
        """
        Return the object represented by the ChoiceParameter option

        :return: the object represented by this option
        """
        return self.qmodel.value()

    def set_value(self, value: Any) -> None:
        """
        Sets the current value for the parameter. The model's value setter is called only if the new value is different
        to avoid emit the changed signal without any reason

        :param value: the new parameter selected key, pointing to the new value
        """
        if len(self.keys):
            if value is None:
                value = self.keys[0]
            if value != self.qmodel.key() and self.validate(value):
                if isinstance(value, (list, dict)):
                    value = json.dumps(value)
                self.qmodel.set_value(value)

    def set_items(self, items: dict[str, object]) -> None:
        """
        Sets the choice items of the ChoiceParameter object. It the object contained items before, these are cleared and
        replaced with the new set.

        :param items: a dictionary containing the items to add to the ChoiceParameter. Key is the name/representation
         of the corresponding option, value is the actual object
        """
        self.qmodel.set_items(items)

    def add_item(self, item: dict[str, object]) -> None:
        """
        Add an item to the ChoiceParameter object

        :param item: the dictionary containing the choice item to add to the ChoiceParameter. Key is the
         name/representation of the corresponding option, value is the actual object.
        """
        self.qmodel.add_item(item)

    def add_items(self, items: dict[str, object]) -> None:
        """
        Add items to the ChoiceParameter object

        :param items: a dictionary containing the choice items to add to the ChoiceParameter. Key is the
         name/representation of the corresponding option, value is the actual object.
        """
        for k, v in items.items():
            self.add_item({k: v})

    @property
    def changed(self) -> SignalInstance:
        return self.qmodel.selection_changed

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, ChoiceParameter):
            return self.key == other.key
        return self.key == other


class Parameters:
    """
    A class to handle a list of parameters
    """

    def __init__(self, parameters: list[Parameter] | Parameter | None = None) -> None:
        if isinstance(parameters, list):
            self.parameter_dict: dict[str, Parameter] = {parameter.name: parameter for parameter in parameters}
        elif isinstance(parameters, Parameter):
            self.parameter_dict: dict[str, Parameter] = {parameters.name: parameters}
        else:
            self.parameter_dict: dict[str, Parameter] = {}
        # if isinstance(parameters, list):
        #     self.parameter_list: list = parameters
        # elif parameters is not None:
        #     self.parameter_list: list = [parameters]
        # else:
        #     self.parameter_list: list = []

    @property
    def parameter_list(self) -> list[Parameter]:
        """
        The list of parameters for legacy
        :return: the parameters as a list
        """
        return list(self.parameter_dict.values())

    @parameter_list.setter
    def parameter_list(self, parameters: list[Parameter] | Parameter | None) -> None:
        if isinstance(parameters, list):
            self.parameter_dict: dict[str, Parameter] = {parameter.name: parameter for parameter in parameters}
        elif isinstance(parameters, Parameter):
            self.parameter_dict: dict[str, Parameter] = {parameters.name: parameters}
        else:
            self.parameter_dict: dict[str, Parameter] = {}

    def update_parameters(self, parameters: list[Parameter] | Parameter, commands: set[str] | None = None) -> None:
        """
        Adds parameters to the list of parameters

        :param commands: list of commands the added parameters are for
        :param parameters: the parameter or list of parameters to add
        """
        if not isinstance(parameters, list):
            parameters = [parameters]
        if commands is not None:
            for p in parameters:
                p.commands.update(commands)
        self.parameter_dict.update({p.name: p for p in parameters})
        for parameter in parameters:
            parameter.reset()

    def reset(self, groups: list[str] | str | None = None) -> None:
        """
        Reset all parameters in groups

        :param groups: the groups to reset
        """
        if self.parameter_list:
            for parameter in self.get_groups(groups):
                parameter.reset()

    def update(self, groups: list[str] | str | None = None) -> None:
        """
        Updates all parameters in groups

        :param groups: the groups to update
        """
        for parameter in self.get_groups(groups):
            parameter.update()

    def values(self, param_list: list[Parameter] | None = None, groups: list[str] | str | None = None) -> dict[str, Parameter]:
        """
        Returns a dictionary of parameters with name as key and value as value

        :param param_list: the list of parameters, if None, all parameters are returned
        :param groups: the parameters groups, if None, all parameters are returned
        :return: a dictionary of all parameters
        """
        if groups is None:
            param_list = param_list or self.parameter_list
        else:
            group_params = self.get_groups(groups)
            param_list = [param for param in (param_list or group_params) if param in group_params]
        return {param.name: param.value for param in param_list}

    def get_groups(self, groups: list[str] | str | None, union: bool = False) -> list[Parameter]:
        """
        Get parameters from groups. If multiple groups are given, either intersection (default) or union of the
        parameters sets is returned

        :param groups: the groups
        :param union: a boolean, if True, a parameter is returned if it belongs to at least one group, if False, it is
         returned only if it belongs to all groups

        :return: a list of parameters
        """
        if groups is None:
            return self.parameter_list
        if isinstance(groups, str):
            groups = {groups}
        if union:
            return [param for param in self.parameter_list if param.groups.union(groups)]
        return [param for param in self.parameter_list if param.groups.intersection(groups)]

    def for_command(self, command: str) -> list[Parameter]:
        """
        Gets all the parameters realted to a specific command

        :param command: the command
        :return: the corresponding list of parameters
        """
        return [param for param in self.parameter_list if command in param.commands]

    def __repr__(self) -> str:
        """
        Return the parameters as a string
        
        :rtype: str
        """
        return f'{self.values()}'

    def __getitem__(self, item: str) -> Parameter:
        """
        Private method enabling Parameters to behave as if it were a dictionary of Parameter objects indexed by their
        name

        :rtype: Parameter
        """
        self_dict = self.to_dict()
        if not isinstance(item, str):
            raise TypeError
        if item in self_dict:
            return self_dict[item]
        raise KeyError

    def __getattr__(self, item: str) -> Parameter:
        """
        Dunder method to allow access to parameters using attribute syntax
        :param item: the name of the parameter
        :return: the parameter
        """
        return self.__getitem__(item)

    def __contains__(self, item: str) -> bool:
        return item in self.to_dict()

    def to_dict(self) -> dict[str, Parameter]:
        """
        Return the parameters as a dictionary

        :return: a dictionary of all parameters, values are Parameter objects
        """
        return {param.name: param for param in self.parameter_list}

    def json(self, param_list: list[Parameter] | None = None, groups: list[str] | str | None = None) -> dict[str, object]:
        """
        Return dictionary of parameters representing each of them as a json-compatible object

        :param param_list: the list of parameters, if None, all parameters are returned
        :param groups: the parameters groups, if None, all parameters are returned
        :return: a dictionary of all parameters in a json-compatible format
        """
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
