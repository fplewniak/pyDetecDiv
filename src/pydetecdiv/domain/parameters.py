"""
Parameter types classes for validation, pre and post-processing of parameters
"""
from pydetecdiv.utils import Singleton


class ParameterFactory(Singleton):
    """
    A factory to create tool parameters objects as defined in the tool XML configuration file.
    """

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

    def create(self, element, tool, **kwargs):
        """
        Create the parameter object corresponding to the type specified in the tool configuration file

        :param element: the parameter element in XML configuration file
        :type element: xml.etree.ElementTree.Element
        :param tool: the tool object
        :type tool: CommandLineTool
        :param kwargs: any extra keyword arguments
        :return: the parameter object
        """
        parameter_type = element.attrib['type'] if 'type' in element.attrib else 'data'
        return self.mapping[parameter_type](element, tool, **kwargs)

    def is_dso(self, type_):
        """
        Return True if this parameter's type corresponds to a Domain-Specific Object

        :param type_: the parameter's type
        :type type_: str
        :return: True if this is a dso, False otherwise
        :rtype: bool
        """
        return self.mapping[type_].is_dso()


class Parameter:
    """
    A generic parameter class to represent both inputs and outputs parameters
    """

    def __init__(self, element, tool, **kwargs):
        self.value = None
        self.dso = None
        self.element = element
        self.tool = tool
        self.type = self.element.attrib['type'] if 'type' in self.element.attrib else 'data'

    def set_value(self, value):
        """
        Set the parameter's value

        :param value: the value
        """
        self.value = value

    def set_dso(self, project):
        """
        Get the DSO or the DSO list corresponding to the parameter's value representing the name of that DSO

        :param project:
        """
        if self.is_dso():
            if self.is_multiple():
                self.dso = [project.get_named_object(self.type, name) for name in self.value]
            else:
                self.dso = project.get_named_object(self.type, self.value)

    @property
    def name(self):
        """
        The parameter's name

        :return: The parameter's name
        :rtype: str
        """
        return self.element.attrib['name']

    @property
    def format(self):
        """
        The data format

        :return: The data format
        :rtype: str
        """
        return self.element.attrib['format'] if self.type == 'data' and 'format' in self.element.attrib else None

    @property
    def label(self):
        """
        The label explaining the parameter for user interaction (GUI, text interface, etc.)

        :return: The label
        :rtype: str
        """
        return self.element.attrib['label'] if 'label' in self.element.attrib else None

    @property
    def default_value(self):
        """
        The default value for this parameter

        :return:The default value
        :rtype: str
        """
        return self.element.attrib['value'] if 'value' in self.element.attrib else None

    def is_input(self):
        """
        Return True if this parameter defines an input

        :return: True if this is an input parameter, False otherwise
        :rtype: bool
        """
        return self.element.tag == 'param'

    def is_output(self):
        """
        Return True if this parameter defines an output

        :return: True if this is an output parameter, False otherwise
        :rtype: bool
        """
        return self.element.tag == 'data'

    def is_multiple(self):
        """
        Return True if this parameter can take multiple values

        :return: True if this parameter can take multiple values, False otherwise
        :rtype: bool
        """
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
        """
        Check the parameter represents a DSO

        :return: True if the parameter represents a DSO, False otherwise
        :rtype: bool
        """
        return False


class TextParameter(Parameter):
    """
    Text parameter
    """
    ...


class IntegerParameter(Parameter):
    """
    Integer parameter
    """

    @property
    def default_value(self):
        """
        The default value for this parameter

        :return:The default value
        :rtype: str
        """
        return int(self.element.attrib['value']) if 'value' in self.element.attrib else 0


class FloatParameter(Parameter):
    """
    Float parameter
    """

    @property
    def default_value(self):
        """
        The default value for this parameter

        :return:The default value
        :rtype: str
        """
        return float(self.element.attrib['value']) if 'value' in self.element.attrib else 0.0


class BooleanParameter(Parameter):
    """
    Boolean parameter
    """

    @property
    def default_value(self):
        """
        The checked value if specified in the configuration file or False if it was not

        :return:The default value
        :rtype: bool
        """
        return self.element.attrib['checked'] in ['True', 'true'] if 'checked' in self.element.attrib else False


class SelectParameter(Parameter):
    """
    Selection parameter
    """
    ...


class ColumnListParameter(Parameter):
    """
    Column list paramter
    """
    ...


class DataParameter(Parameter):
    """
    Data parameter
    """

    def is_image(self):
        return self.format in ['imagetiff']


class DataCollectionParameter(Parameter):
    """
    Data collection parameter
    """
    ...


class DirectoryUriParameter(Parameter):
    """
    Directory parameter
    """
    ...


class FovParameter(Parameter):
    """
    Parameter representing one or several FOVs
    """

    @staticmethod
    def is_dso():
        """
        Returns True since this is a DSO

        :return: True
        """
        return True


class RoiParameter(Parameter):
    """
    Parameter representing one or several ROIs
    """

    @staticmethod
    def is_dso():
        """
        Returns True since this is a DSO

        :return: True
        """
        return True


class DatasetParameter(Parameter):
    """
    Parameter representing a Dataset
    """

    @staticmethod
    def is_dso():
        """
        Returns True since this is a DSO

        :return: True
        """
        return True
