"""
Parameter widget classes to automatically create forms for launching tools and plugins
"""
from PySide6.QtWidgets import QFrame, QLabel, QLineEdit, QSpinBox, QDoubleSpinBox, QCheckBox, QComboBox, \
    QListWidget, QAbstractItemView

from pydetecdiv.app import pydetecdiv_project, PyDetecDiv
from pydetecdiv.utils import singleton


@singleton
class ParameterWidgetFactory:
    """
    A factory to create widgets ToolForm for tool parameters according to the tool XML configuration file.
    """

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
        """
        Create the widget suitable for the parameter according to its type
        :param parameter: the parameter object
        :type parameter: Parameter
        :param kwargs: any extra keywords argument passed to the parameter widget object
        :return: the widget to include into the tool form
        :rtype: ParameterWidget
        """
        return self.mapping[parameter.type](parameter, **kwargs)


class Options:
    """
    The options for a selection-type widget (including dynamic parameters for DSO selection, etc)
    """

    def __init__(self, parameter):
        self.parameter = parameter
        self.data_type = 'select'
        self.filters = []

    @property
    def list(self):
        """
        The list of options to select from
        :return: dictionary of options
        """
        option_list = self.parameter.element.findall('.//option')
        options = self.parameter.element.find('.//options')
        if option_list:
            return {o.text: Option(o.text, **o.attrib) for o in option_list}
        if options:
            return self.get_option_list(options)
        return None

    def get_option_list(self, options):
        """
        Determine the list of options when they are defined by OPTIONS tag, i.e. dynamic options as opposed to the
        static options defined by OPTION tag
        :param options: the OPTIONS element
        :type: xml.etree.ElementTree.Element
        :return: dictionary of options
        """
        if 'from_data_table' in options.attrib:
            self.data_type = options.attrib['from_data_table']
            columns = self.get_columns(options)
            if columns:
                with pydetecdiv_project(PyDetecDiv().project_name) as project:
                    dso_list = project.get_objects(self.data_type)
                    return {dso.record()[columns['value']]: Option(dso.record()[columns['value']]) for dso in dso_list}
        else:
            for filter_ in self.get_filters(options):
                match filter_.element.attrib['type']:
                    case 'data_meta':
                        ref_obj = self.parameter.tool.parameters[filter_.element.attrib['ref']].dso
                        if ref_obj:
                            return getattr(ref_obj, filter_.element.attrib["key"])
                        return {}
                        # return {filter_.element.attrib['ref']: filter_.element.attrib['key']}
        return {}

    def get_columns(self, options):
        """
        Get the columns defined in options
        :param options: the OPTIONS element
        :type: xml.etree.ElementTree.Element
        :return: dictionary of columns
        """
        columns = options.findall('.//column')
        return {c.attrib['name']: c.attrib['index'] for c in columns} if columns else {}

    def get_filters(self, options):
        """
        Get the filters defined in options
        :param options: the OPTIONS element
        :type: xml.etree.ElementTree.Element
        :return: list of filters
        """
        filters = options.findall('.//filter')
        return [Filter(f) for f in filters] if filters else []


class Option:
    """
    An option with its value and text to display in the selection widget, and any extra attributes
    """

    def __init__(self, text_value, **kwargs):
        self.value = kwargs['value'] if 'value' in kwargs else text_value
        self.text_value = text_value
        self.attrib = kwargs

    def __dict__(self):
        return {'text': self.text_value}.update(self.attrib)


class Filter:
    """
    A filter used to determine the items in the selection widget
    """

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
        """
        Set the value of the parameter to whatever the user has chosen in the ToolForm
        """
        self.parameter.set_value(self.get_value())

    def get_value(self):
        """
        A generic abstract method to return the user-defined value of the parameter. This method must be implemented by
        type-specific parameter widget classes
        """
        raise NotImplementedError


class TextParameterWidget(ParameterWidget):
    """
    A widget for text input
    """

    def __init__(self, parameter, parent=None, layout=None, **kwargs):
        super().__init__(parameter, parent=parent, layout=layout, **kwargs)
        self.value = QLineEdit(parent=self)
        self.value.setText(self.parameter.default_value)
        self.layout.addRow(QLabel(self.parameter.label), self.value)

    def get_value(self):
        """
        Return the text value in the text parameter widget
        :return: text value
        :rtype: str
        """
        return self.value.text()


class IntegerParameterWidget(ParameterWidget):
    """
    A spin box widget for integer input
    """

    def __init__(self, parameter, parent=None, layout=None, **kwargs):
        super().__init__(parameter, parent=parent, layout=layout, **kwargs)
        self.spin_box = QSpinBox(parent=self)
        self.spin_box.setValue(self.parameter.default_value)
        self.layout.addRow(QLabel(self.parameter.label), self.spin_box)

    def get_value(self):
        """
        Return the integer value specified by the spin box
        :return: integer value
        :rtype: int
        """
        return self.spin_box.value()


class FloatParameterWidget(ParameterWidget):
    """
    A spin box widget for float input
    """

    def __init__(self, parameter, parent=None, layout=None, **kwargs):
        super().__init__(parameter, parent=parent, layout=layout, **kwargs)
        self.spin_box = QDoubleSpinBox(parent=self)
        self.spin_box.setValue(self.parameter.default_value)
        self.layout.addRow(QLabel(self.parameter.label), self.spin_box)

    def get_value(self):
        """
        Return the float value specified by the spin box
        :return: float value
        :rtype: float
        """
        return self.spin_box.value()


class BooleanParameterWidget(ParameterWidget):
    """
    A check box widget for boolean input
    """

    def __init__(self, parameter, parent=None, layout=None, **kwargs):
        super().__init__(parameter, parent=parent, layout=layout, **kwargs)
        self.check_box = QCheckBox(parent=self)
        self.check_box.setChecked(self.parameter.default_value)
        self.layout.addRow(QLabel(self.parameter.label), self.check_box)

    def get_value(self):
        """
        Return the checked status of the check box
        :return: boolean value
        :rtype: bool
        """
        return self.check_box.isChecked()


class SelectParameterWidget(ParameterWidget):
    """
    A selection widget (QComboBox)
    """

    def __init__(self, parameter, parent=None, layout=None, **kwargs):
        super().__init__(parameter, parent=parent, layout=layout, **kwargs)
        self.combo = QComboBox(parent=self)
        for text, _ in self.options.items():
            self.combo.addItem(text)
        self.layout.addRow(QLabel(self.parameter.label), self.combo)

    def get_value(self):
        """
        Return the selection
        :return: selected value
        :rtype: object
        """
        return self.options[self.combo.currentText()].value


class ColumnListParameterWidget(ParameterWidget):
    """

    """
    ...


class DataParameterWidget(ParameterWidget):
    """

    """

    def __init__(self, parameter, parent=None, layout=None, **kwargs):
        super().__init__(parameter, parent=parent, layout=layout, **kwargs)

    def get_value(self):
        ...


class DataCollectionParameterWidget(ParameterWidget):
    """

    """
    ...


class DirectoryUriParameterWidget(ParameterWidget):
    """

    """
    ...


class FovParameterWidget(ParameterWidget):
    """
    A selection widget to select FOVs
    """

    def __init__(self, parameter, parent=None, layout=None, **kwargs):
        super().__init__(parameter, parent=parent, layout=layout, **kwargs)
        if self.parameter.is_multiple():
            self.fov_list = QListWidget(parent=self)
            self.fov_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        else:
            self.fov_list = QComboBox(parent=self)
        with pydetecdiv_project(PyDetecDiv().project_name) as project:
            if project.count_objects('FOV'):
                fov_list = [fov.name for fov in project.get_objects('FOV')]
                self.fov_list.addItems(sorted(fov_list))
        self.layout.addRow(QLabel(self.parameter.label), self.fov_list)

    def get_value(self):
        """
        Return the FOV selection
        :return: selected FOV(s)
        :rtype: list of FOVs or FOV
        """
        if self.parameter.is_multiple():
            return [item.text() for item in self.fov_list.selectedItems()]
        return self.fov_list.currentText()


class RoiParameterWidget(ParameterWidget):
    """
    A selection widget to select ROIs
    """

    def __init__(self, parameter, parent=None, layout=None, **kwargs):
        super().__init__(parameter, parent=parent, layout=layout, **kwargs)
        self.options = Options(self.parameter).list
        if self.parameter.is_multiple():
            self.roi_list = QListWidget(parent=self)
            self.roi_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        else:
            self.roi_list = QComboBox(parent=self)
        with pydetecdiv_project(PyDetecDiv().project_name) as project:
            if project.count_objects('ROI'):
                roi_list = [roi.name for roi in project.get_objects('ROI')]
                self.roi_list.addItems(sorted(roi_list))
        self.layout.addRow(QLabel(self.parameter.label), self.roi_list)

    def get_value(self):
        """
        Return the ROI selection
        :return: selected ROI(s)
        :rtype: list of ROIs or ROI
        """
        if self.parameter.is_multiple():
            return [item.text() for item in self.roi_list.selectedItems()]
        return self.roi_list.currentText()


class DatasetParameterWidget(ParameterWidget):
    """

    """
    ...
