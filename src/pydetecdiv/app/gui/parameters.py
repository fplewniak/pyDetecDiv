"""
Parameter widget classes to automatically create forms for launching tools and plugins
"""
from PySide6.QtWidgets import QFrame, QLabel, QLineEdit, QSpinBox, QDoubleSpinBox, QCheckBox, QComboBox, \
    QListWidget, QAbstractItemView

from pydetecdiv.app import pydetecdiv_project, PyDetecDiv
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

    def __init__(self, parameter, parent=None, layout=None, **kwargs):
        super().__init__(parent)
        self.parameter = parameter
        self.layout = layout

    def set_value(self):
        self.parameter.set_value(self.get_value())

    def get_value(self):
        ...


class TextParameterWidget(ParameterWidget):
    def __init__(self, parameter, parent=None, layout=None, **kwargs):
        super().__init__(parameter, parent=parent, layout=layout, **kwargs)
        self.value = QLineEdit(parent=self)
        self.layout.addRow(QLabel(self.parameter.label), self.value)

    def get_value(self):
        return self.value.text()


class IntegerParameterWidget(ParameterWidget):
    def __init__(self, parameter, parent=None, layout=None, **kwargs):
        super().__init__(parameter, parent=parent, layout=layout, **kwargs)
        self.spin_box = QSpinBox(parent=self)
        self.layout.addRow(QLabel(self.parameter.label), self.spin_box)

    def get_value(self):
        return self.spin_box.value()


class FloatParameterWidget(ParameterWidget):
    def __init__(self, parameter, parent=None, layout=None, **kwargs):
        super().__init__(parameter, parent=parent, layout=layout, **kwargs)
        self.spin_box = QDoubleSpinBox(parent=self)
        self.layout.addRow(QLabel(self.parameter.label), self.spin_box)

    def get_value(self):
        return self.spin_box.value()


class BooleanParameterWidget(ParameterWidget):
    def __init__(self, parameter, parent=None, layout=None, **kwargs):
        super().__init__(parameter, parent=parent, layout=layout, **kwargs)
        self.check_box = QCheckBox(parent=self)
        self.layout.addRow(QLabel(self.parameter.label), self.check_box)

    def get_value(self):
        return self.check_box.isChecked()


class SelectParameterWidget(ParameterWidget):
    def __init__(self, parameter, parent=None, layout=None, **kwargs):
        super().__init__(parameter, parent=parent, layout=layout, **kwargs)
        self.combo = QComboBox(parent=self)
        for text, value in self.parameter.options.items():
            self.combo.addItem(text)
        self.layout.addRow(QLabel(self.parameter.label), self.combo)

    def get_value(self):
        return self.parameter.options[self.combo.currentText()]


class ColumnListParameterWidget(ParameterWidget):
    ...


class DataParameterWidget(ParameterWidget):
    def __init__(self, parameter, parent=None, layout=None, **kwargs):
        super().__init__(parameter, parent=parent, layout=layout, **kwargs)


class DataCollectionParameterWidget(ParameterWidget):
    ...


class DirectoryUriParameterWidget(ParameterWidget):
    ...


class FovParameterWidget(ParameterWidget):
    def __init__(self, parameter, parent=None, layout=None, **kwargs):
        super().__init__(parameter, parent=parent, layout=layout, **kwargs)
        if self.parameter.is_multiple():
            self.fov_list = QListWidget(parent=self)
            print(self.fov_list.selectionMode())
            self.fov_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        else:
            self.fov_list = QComboBox(parent=self)
        with pydetecdiv_project(PyDetecDiv().project_name) as project:
            if project.count_objects('FOV'):
                FOV_list = [fov.name for fov in project.get_objects('FOV')]
                self.fov_list.addItems(sorted(FOV_list))
        self.layout.addRow(QLabel(self.parameter.label), self.fov_list)

    def get_value(self):
        if self.parameter.is_multiple():
            return [item.text() for item in self.fov_list.selectedItems()]
        return self.fov_list.currentText()


class RoiParameterWidget(ParameterWidget):
    ...


class DatasetParameterWidget(ParameterWidget):
    ...
