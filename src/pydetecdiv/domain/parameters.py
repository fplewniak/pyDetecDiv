"""
Parameter types classes for validation, pre and post processing of parameters
"""
from pydetecdiv.app import pydetecdiv_project, PyDetecDiv
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

    def create_from_dict(self, name, type_, **kwargs):
        return self.mapping[type_](name, type_, **kwargs)

    def create(self, element, **kwargs):
        parameter_type = element.attrib['type'] if 'type' in element.attrib else 'data'
        return self.mapping[parameter_type](element, **kwargs)

    def is_dso(self, type_):
        return self.mapping[type_].is_dso()


class Parameter:
    """
    A generic parameter class to represent both inputs and outputs parameters
    """

    def __init__(self, element, **kwargs):
        self.value = None
        self.dso = None
        self.element = element

    def set_value(self, value):
        self.value = value

    def set_dso(self, project):
        if self.is_dso():
            if self.is_multiple():
                self.dso = [project.get_named_object(self.type, name) for name in self.value]
            else:
                self.dso = project.get_named_object(self.type, self.value)

    @property
    def options(self):
        return {o.text: o.attrib['value'] for o in self.element.findall('.//option')}

    @property
    def name(self):
        return self.element.attrib['name']

    @property
    def type(self):
        return self.element.attrib['type'] if 'type' in self.element.attrib else 'data'

    @property
    def format(self):
        return self.element.attrib['format'] if self.type == 'data' and 'format' in self.element.attrib else None

    @property
    def label(self):
        return self.element.attrib['label'] if 'label' in self.element.attrib else None

    @property
    def default_value(self):
        return self.element.attrib['value'] if 'value' in self.element.attrib else None

    def is_input(self):
        return self.element.tag == 'param'

    def is_output(self):
        return self.element.tag == 'data'

    def is_multiple(self):
        return self.element.attrib['multiple'] in ['True', 'true'] if 'multiple' in self.element.attrib else False

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
    def __init__(self, element, **kwargs):
        super().__init__(element, **kwargs)


class IntegerParameter(Parameter):
    def __init__(self, element, **kwargs):
        super().__init__(element, **kwargs)


class FloatParameter(Parameter):
    def __init__(self, element, **kwargs):
        super().__init__(element, **kwargs)


class BooleanParameter(Parameter):
    def __init__(self, element, **kwargs):
        super().__init__(element, **kwargs)

    @property
    def default_value(self):
        return self.element.attrib['checked'] in ['True', 'true'] if 'checked' in self.element.attrib else False


class SelectParameter(Parameter):
    def __init__(self, element, **kwargs):
        super().__init__(element, **kwargs)


class ColumnListParameter(Parameter):
    def __init__(self, element, **kwargs):
        super().__init__(element, **kwargs)


class DataParameter(Parameter):
    def __init__(self, element, **kwargs):
        super().__init__(element, **kwargs)

    def is_image(self):
        return self.format in ['imagetiff']


class DataCollectionParameter(Parameter):
    def __init__(self, element, **kwargs):
        super().__init__(element, **kwargs)


class DirectoryUriParameter(Parameter):
    def __init__(self, element, **kwargs):
        super().__init__(element, **kwargs)


class FovParameter(Parameter):
    def __init__(self, element, **kwargs):
        super().__init__(element, **kwargs)

    @staticmethod
    def is_dso():
        return True


class RoiParameter(Parameter):
    def __init__(self, element, **kwargs):
        super().__init__(element, **kwargs)

    @staticmethod
    def is_dso():
        return True


class DatasetParameter(Parameter):
    def __init__(self, element, **kwargs):
        super().__init__(element, **kwargs)

    @staticmethod
    def is_dso():
        return True
