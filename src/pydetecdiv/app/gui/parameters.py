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


class Options:
    def __init__(self, parameter):
        self.parameter = parameter
        self.data_type = 'select'
        self.filters = []

    @property
    def list(self):
        option_list = self.parameter.element.findall('.//option')
        options = self.parameter.element.find('.//options')
        if option_list:
            return {o.text: Option(o.text, **o.attrib) for o in option_list}
        elif options:
            return self.get_option_list(options)
        return None

    def get_option_list(self, options):
        if 'from_data_table' in options.attrib:
            self.data_type = options.attrib['from_data_table']
            columns = self.get_columns(options)
            if columns:
                with pydetecdiv_project(PyDetecDiv().project_name) as project:
                    dso_list = project.get_objects(self.data_type)
                    return {dso.record()[columns['value']]: Option(dso.record()[columns['value']]) for dso in dso_list}
        else:
            for filter in self.get_filters(options):
                match filter.element.attrib['type']:
                    case 'data_meta':
                        ref_obj = self.parameter.tool.parameters[filter.element.attrib['ref']].dso
                        if ref_obj:
                            return eval(f'{ref_obj}.{filter.element.attrib["key"]}')
                        return {}
                        # return {filter.element.attrib['ref']: filter.element.attrib['key']}
        return {}

    def get_columns(self, options):
        columns = options.findall('.//column')
        return {c.attrib['name']: c.attrib['index'] for c in columns} if columns else {}

    def get_filters(self, options):
        filters = options.findall('.//filter')
        return [Filter(f) for f in filters] if filters else []


class Option:
    def __init__(self, text_value, **kwargs):
        self.value = kwargs['value'] if 'value' in kwargs else text_value
        self.text_value = text_value
        self.attrib = kwargs

    def __dict__(self):
        return {'text': self.text_value}.update(self.attrib)


class Filter:
    def __init__(self, element):
        self.element = element


class ParameterWidget(QFrame):
    """
    A generic parameter class to represent both inputs and outputs parameters
    """

    def __init__(self, parameter, parent=None, layout=None, **kwargs):
        super().__init__(parent)
        self.parameter = parameter
        self.layout = layout
        self.options = Options(self.parameter).list

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
        self.check_box.setChecked(self.parameter.default_value)
        self.layout.addRow(QLabel(self.parameter.label), self.check_box)

    def get_value(self):
        return self.check_box.isChecked()


class SelectParameterWidget(ParameterWidget):
    def __init__(self, parameter, parent=None, layout=None, **kwargs):
        super().__init__(parameter, parent=parent, layout=layout, **kwargs)
        self.combo = QComboBox(parent=self)
        for text, value in self.options.items():
            self.combo.addItem(text)
        self.layout.addRow(QLabel(self.parameter.label), self.combo)

    def get_value(self):
        return self.options[self.combo.currentText()].value


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
    def __init__(self, parameter, parent=None, layout=None, **kwargs):
        super().__init__(parameter, parent=parent, layout=layout, **kwargs)
        self.options = Options(self.parameter).list
        print(self.options)
        if self.parameter.is_multiple():
            self.roi_list = QListWidget(parent=self)
            self.roi_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        else:
            self.roi_list = QComboBox(parent=self)
        with pydetecdiv_project(PyDetecDiv().project_name) as project:
            if project.count_objects('ROI'):
                ROI_list = [roi.name for roi in project.get_objects('ROI')]
                self.roi_list.addItems(sorted(ROI_list))
        self.layout.addRow(QLabel(self.parameter.label), self.roi_list)

    def get_value(self):
        if self.parameter.is_multiple():
            return [item.text() for item in self.roi_list.selectedItems()]
        return self.roi_list.currentText()


class DatasetParameterWidget(ParameterWidget):
    ...
