"""
Parameter widget classes to automatically create forms for launching tools and plugins
"""
from PySide6.QtWidgets import QFrame

from pydetecdiv.utils import singleton


@singleton
class ParameterWidgetFactory:
    def __init__(self):
        self.mapping = {
            'text': TextParameterWidget,
            'integer': IntegerParameterWidget,
            'float': FloatParameterWidget,
            'boolean': BooleanParameterWidget,
            'select': SelectParameterWidget,
            'data_column': ColumnListParameterWidget,
            'data': DataParameterWidget,
            'data_collection': DataCollectionParameterWidget,
            'directory_uri': DirectoryUriParameterWidget,
            'FOV': FovParameterWidget,
            'ROI': RoiParameterWidget,
            'Dataset': DatasetParameterWidget
        }

    def create(self, parameter, **kwargs):
        return self.mapping[parameter.type](parameter, **kwargs)


class ParameterWidget(QFrame):
    """
    A generic parameter class to represent both inputs and outputs parameters
    """

    def __init__(self, parameter, parent=None, **kwargs):
        super().__init__(parent)
        self.parameter = parameter


class TextParameterWidget(ParameterWidget):
    ...


class IntegerParameterWidget(ParameterWidget):
    ...


class FloatParameterWidget(ParameterWidget):
    ...


class BooleanParameterWidget(ParameterWidget):
    ...


class SelectParameterWidget(ParameterWidget):
    ...


class ColumnListParameterWidget(ParameterWidget):
    ...


class DataParameterWidget(ParameterWidget):
    ...


class DataCollectionParameterWidget(ParameterWidget):
    ...


class DirectoryUriParameterWidget(ParameterWidget):
    ...


class FovParameterWidget(ParameterWidget):
    ...


class RoiParameterWidget(ParameterWidget):
    ...


class DatasetParameterWidget(ParameterWidget):
    ...
