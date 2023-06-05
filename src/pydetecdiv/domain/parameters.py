"""
Parameter types classes for validation, pre and post processing of parameters
"""
from pydetecdiv.utils import singleton


@singleton
class ParameterFactory:
    def __init__(self):
        self.mapping = {
            'text': TextParameter,
            'integer': IntegerParameter,
            'float': FloatParameter,
            'boolean': BooleanParameter,
            'select': SelectParameter,
            'data_column': ColumnListParameter,
            'data': DataParameter,
            'data_collection': DataCollectionParameter,
            'directory_uri': DirectoryUriParameter,
            'FOV': FovParameter,
            'ROI': RoiParameter,
            'Dataset': DatasetParameter
        }

    def create(self, name, type_, **kwargs):
        return self.mapping[type_](name, type_, **kwargs)

    def is_dso(self, type_):
        return self.mapping[type_].is_dso()


class Parameter:
    """
    A generic parameter class to represent both inputs and outputs parameters
    """

    def __init__(self, name, type_, **kwargs):
        self.name = name
        self.type = type_
        self.format = kwargs['format'] if type_ == 'data' and 'format' in kwargs else None
        self.label = kwargs['label'] if 'label' in kwargs else None
        self.value = None
        self.obj = None

    def is_image(self):
        """
        Check the parameter represents image data
        :return: True if the parameter represents image data, False otherwise
        :rtype: bool
        """
        return False

    @staticmethod
    def is_dso():
        return False


class TextParameter(Parameter):
    def __init__(self, name, type_, **kwargs):
        super().__init__(name, type_, **kwargs)


class IntegerParameter(Parameter):
    def __init__(self, name, type_, **kwargs):
        super().__init__(name, type_, **kwargs)


class FloatParameter(Parameter):
    def __init__(self, name, type_, **kwargs):
        super().__init__(name, type_, **kwargs)


class BooleanParameter(Parameter):
    def __init__(self, name, type_, **kwargs):
        super().__init__(name, type_, **kwargs)


class SelectParameter(Parameter):
    def __init__(self, name, type_, multiple=False, **kwargs):
        super().__init__(name, type_, **kwargs)
        print(f'{kwargs}')


class ColumnListParameter(Parameter):
    def __init__(self, name, type_, **kwargs):
        super().__init__(name, type_, **kwargs)


class DataParameter(Parameter):
    def __init__(self, name, type_, **kwargs):
        super().__init__(name, type_, **kwargs)

    def is_image(self):
        return self.format in ['imagetiff']


class DataCollectionParameter(Parameter):
    def __init__(self, name, type_, **kwargs):
        super().__init__(name, type_, **kwargs)


class DirectoryUriParameter(Parameter):
    def __init__(self, name, type_, **kwargs):
        super().__init__(name, type_, **kwargs)


class FovParameter(Parameter):
    def __init__(self, name, type_, **kwargs):
        super().__init__(name, type_, **kwargs)

    @staticmethod
    def is_dso():
        return True


class RoiParameter(Parameter):
    def __init__(self, name, type_, **kwargs):
        super().__init__(name, type_, **kwargs)

    @staticmethod
    def is_dso():
        return True


class DatasetParameter(Parameter):
    def __init__(self, name, type_, **kwargs):
        super().__init__(name, type_, **kwargs)

    @staticmethod
    def is_dso():
        return True
