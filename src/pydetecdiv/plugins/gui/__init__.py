"""
Module defining widgets and other utilities for creating windows/forms with a minimum of code
"""
import json

from PySide6.QtCore import QStringListModel
from PySide6.QtGui import QIcon
from PySide6.QtSql import QSqlQueryModel
from PySide6.QtWidgets import QDialog, QVBoxLayout, QGroupBox, QFormLayout, QLabel, QDialogButtonBox, \
    QSizePolicy, QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox, QAbstractSpinBox, QTableView, QAbstractItemView, \
    QPushButton, QApplication, QRadioButton, QListView


# class ParameterWidgets:
#     """
#     A class to handle plugin parameters from a Form gui
#     """
#
#     def __init__(self):
#         self.param_groups = {}
#
#     def add_groups(self, groups):
#         """
#         Add empty groups of parameters
#
#         :param groups: the list of groups
#         """
#         for group in groups:
#             self.add_group(group)
#
#     def add_group(self, group, parameters=None):
#         """
#         Add a new group of parameters (in a dictionary)
#
#         :param group: the group to create
#         :param param_dict: the dictionary of parameters
#         """
#         self.param_groups[group] = parameters if parameters is not None else {}
#
#     def add(self, group, parameter):
#         """
#         Add parameters (in a dictionary) to an existing group of parameters
#
#         :param group: the group of parameters to expand
#         :param param_dict: the parameters
#         """
#         self.param_groups[group].update(parameter)
#
#     def get_values(self, group):
#         """
#         Get a dictionary containing all parameters key/values for a given group
#
#         :param group: the requested parameter group
#         :return: a dictionary of parameters
#         """
#         return {name: widget.value() for name, widget in self.param_groups[group].items()}
#
#     def get_value(self, name, group):
#         return self.get_values(group)[name]


class StyleSheets:
    """
    Style sheets for the widgets
    """
    groupBox = """
                QGroupBox {
                    border: 1px solid lightgray;
                    border-radius: 3px;
                    padding-top: 0.5em;

                    padding-bottom: 0.5em;
                    margin-top: 0.5em;
                    font-weight: bold;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    subcontrol-position: top center;
                }
                """


class GroupBox(QGroupBox):
    """
    an extension of QGroupBox class
    """

    def __init__(self, parent, title=None):
        super().__init__(parent)
        if title is not None:
            self.setTitle(title)
        self.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Maximum)
        self.layout = self.layout()

    @property
    def plugin(self):
        """
        Property returning the plugin from the top parent of the current widget

        :return: the plugin module or None if it was not found
        """
        parent = self.parent()
        while parent:
            if hasattr(parent, 'plugin'):
                return parent.plugin
        return None

    def addSubBox(self, widget, **kwargs):
        sub_box = widget(self, **kwargs)
        self.layout.addWidget(sub_box)
        return sub_box


class ParametersFormGroupBox(GroupBox):
    """
    an extension of GroupBox class to handle Forms
    """

    def __init__(self, parent, title=None, show=True):
        super().__init__(parent, title)
        self.layout = QFormLayout(self)
        self.setLayout(self.layout)
        self.setVisible(show)

    def addSubBox(self, widget, **kwargs):
        sub_box = widget(self, **kwargs)
        self.layout.addRow(sub_box)
        return sub_box

    def addOption(self, label=None, widget=None, parameter=None, **kwargs):
        """
        add an option to the current Form

        :param label: the label for the option
        :param widget: the widget to specify the option value, etc
        :param kwargs: extra args passed to the widget
        :return: the option widget
        """
        if issubclass(widget, (QPushButton, QDialogButtonBox)):
            option = widget(self, **kwargs)
        else:
            option = widget(self, parameter, **kwargs)
            option.changed.connect(parameter.set_value)
            parameter.changed.connect(option.setValue)
            parameter.reset()
        # if parameter is not None:
        #     groups, param = parameter
        #     if not isinstance(groups, list):
        #         groups = [groups]
        #     for group in groups:
        #         self.plugin.parameter_widgets.add(group, {parameter: option})

        if label is None:
            self.layout.addRow(option)
        else:
            self.layout.addRow(QLabel(label), option)
        return option

    def setRowVisible(self, index, on=True):
        """
        set the row defined by index or widget visible or invisible in the form layout

        :param index: the row index or the widget in that row
        :param on: whether the row should be visible or not
        """
        self.layout.setRowVisible(index, on)


class ComboBox(QComboBox):
    """
    an extension of the QComboBox class
    """

    def __init__(self, parent, parameter, editable=False):
        super().__init__(parent)
        if parameter.items is not None:
            self.addItemDict(parameter.items)
        self.setEditable(editable)

    def addItemDict(self, options):
        """
        add items to the ComboBox as a dictionary

        :param options: dictionary of options specifying labels and corresponding user data {label: userData, ...}
        """
        for label, data in options.items():
            self.addItem(label, userData=data)

    def setText(self, text):
        if self.findText(text) != -1:
            self.setCurrentText(text)
        else:
            self.addItem(text)

    def text(self):
        return self.currentText()

    @property
    def selected(self):
        """
        return property telling whether the current index of this ComboBox has changed

        :return: boolean indication whether the current index has changed (i.e. new selection)
        """
        return self.currentIndexChanged

    @property
    def changed(self):
        """
        return property telling whether the current text of this ComboBox has changed. This overwrites the Pyside
        equivalent method in order to have the same method name for all widgets

        :return: boolean indication whether the current text has changed (i.e. new selection)
        """
        return self.currentTextChanged

    def value(self):
        """
        method to standardize the way widget values from a form are returned

        :return: the current data (if it can be json serialized) or the current text of the selected item
        """
        if self.currentData() is not None:
            try:
                _ = json.dumps(self.currentData())
                return self.currentData()
            except TypeError:
                pass
        try:
            return json.loads(self.currentText())
        except json.decoder.JSONDecodeError:
            return self.currentText()

    def setValue(self, value):
        self.setCurrentText(value)


class ListView(QListView):
    """
    an extension of the QComboBox class
    """

    def __init__(self, parent, parameter, height=None, multiselection=False, **kwargs):
        super().__init__(parent, **kwargs)
        if multiselection:
            self.setSelectionMode(QAbstractItemView.MultiSelection)
        if height is not None:
            self.setFixedHeight(height)
        self.setModel(QStringListModel())
        self.items = parameter.items
        if parameter.items is not None:
            self.addItemDict(parameter.items)

    def addItemDict(self, options):
        """
        add items to the ComboBox as a dictionary

        :param options: dictionary of options specifying labels and corresponding user data {label: userData, ...}
        """
        self.items = options
        self.model().setStringList(list(options.keys()))

    # def setText(self, text):
    #     if self.findText(text) != -1:
    #         self.setCurrentText(text)
    #     else:
    #         self.addItem(text)
    #
    # def text(self):
    #     return self.currentText()
    #
    # @property
    # def selected(self):
    #     """
    #     return property telling whether the current index of this ComboBox has changed
    #
    #     :return: boolean indication whether the current index has changed (i.e. new selection)
    #     """
    #     return self.currentIndexChanged
    #
    # @property
    # def changed(self):
    #     """
    #     return property telling whether the current text of this ComboBox has changed. This overwrites the Pyside
    #     equivalent method in order to have the same method name for all widgets
    #
    #     :return: boolean indication whether the current text has changed (i.e. new selection)
    #     """
    #     return self.currentTextChanged
    #
    def selection(self):
        """
        method to standardize the way widget values from a form are returned

        :return: the current data (if it is defined) or the current text of the selected item
        """
        return [self.items[self.model().data(idx)] for idx in
                sorted(self.selectedIndexes(), key=lambda x: x.row(), reverse=False)]


class LineEdit(QLineEdit):
    """
    an extension of QLineEdit class
    """

    def __init__(self, parent, parameter, editable=True):
        super().__init__(parent)
        self.setEditable(editable)

    def value(self):
        """
        method to standardize the way widget values from a form are returned

        :return: the text in LineEdit
        """
        try:
            return json.loads(self.text())
        except json.decoder.JSONDecodeError:
            return self.text()

    def setValue(self, value):
        self.setText(value)

    @property
    def changed(self):
        return self.textChanged

    @property
    def edited(self):
        return self.editingFinished

    def setEditable(self, editable=True):
        self.setReadOnly(not editable)


class PushButton(QPushButton):
    """
    an extension of QPushButton class
    """

    def __init__(self, parent, text, icon=None, flat=False):
        if icon is None:
            super().__init__(text, parent)
        else:
            super().__init__(icon, text, parent)
        self.setFlat(flat)


class AdvancedButton(PushButton):
    """
    an extension of PushButton class to control collapsible group boxes for advanced options
    """

    def __init__(self, parent):
        super().__init__(parent, text='Advanced options', icon=QIcon(':icons/show'), flat=True)
        self.group_box = None
        self.clicked.connect(self.toggle)

    def hide(self):
        """
        hide the linked group box
        """
        super().hide()
        self.setIcon(QIcon(':icons/show'))
        self.group_box.setVisible(False)

    def linkGroupBox(self, group_box):
        """
        link this advanced button to a group box whose expansion or collapse should be controlled by this button

        :param group_box: the group box to link to this button
        """
        self.group_box = group_box

    def toggle(self):
        """
        toggle the advanced button to and from show/hide form
        """
        if self.group_box.isVisible():
            self.setIcon(QIcon(':icons/show'))
            self.group_box.setVisible(False)
        else:
            self.setIcon(QIcon(':icons/hide'))
            self.group_box.setVisible(True)
        self.parent().parent().fit_to_contents()


class RadioButton(QRadioButton):
    """
    an extension of the QRadioButton class
    """

    def __init__(self, parent, parameter, exclusive=True):
        super().__init__(None, parent)
        self.setAutoExclusive(exclusive)

    def value(self):
        """
        method to standardize the way widget values from a form are returned

        :return: boolean, True if button is checked, False otherwise
        """
        return self.isChecked()


class SpinBox(QSpinBox):
    """
    an extension of the QSpinBox class
    """

    def __init__(self, parent, parameter, range=(1, 4096), single_step=1, adaptive=False):
        super().__init__(parent)
        self.setRange(*range[0:2])
        self.setSingleStep(single_step)
        if adaptive:
            self.setStepType(QAbstractSpinBox.AdaptiveDecimalStepType)
        self.setValue(parameter.value)

    @property
    def changed(self):
        """
        return property telling whether the spinbox value has changed. This overwrites the Pyside equivalent method in
         order to have the same method name for all widgets

        :return: boolean indication whether the value has changed
        """
        return self.valueChanged


class DoubleSpinBox(QDoubleSpinBox):
    """
    an extension of the QDoubleSpinBox class
    """

    def __init__(self, parent, parameter, range=(0.1, 1.0), decimals=2, single_step=0.1, adaptive=False, value=0.1,
                 enabled=True):
        super().__init__(parent)
        self.setRange(*range[0:2])
        self.setDecimals(decimals)
        self.setSingleStep(single_step)
        if adaptive:
            self.setStepType(QAbstractSpinBox.AdaptiveDecimalStepType)
        if value is None:
            self.setValue(range[0])
        else:
            self.setValue(value)
        self.setEnabled(enabled)

    @property
    def changed(self):
        """
        method returning whether the spinbox value has been changed

        :return: boolean, True if the value was changed, False otherwise
        """
        return self.valueChanged


class TableView(QTableView):
    """
    an extension of the QTableView widget
    """

    def __init__(self, parent, parameter, multiselection=True, behavior='rows'):
        super().__init__(parent)
        self.model = QSqlQueryModel()
        self.setModel(self.model)
        if multiselection:
            self.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        match behavior:
            case 'rows':
                self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
            case _:
                pass

    def setQuery(self, query):
        """
        set the SQL query used to feed the model of the table view

        :param query: the query
        """
        self.model.setQuery(query)


class DialogButtonBox(QDialogButtonBox):
    """
    A extension of QDialogButtonBox to add a button box
    """

    def __init__(self, parent, buttons=(QDialogButtonBox.Ok, QDialogButtonBox.Close)):
        super().__init__(parent)
        for button in buttons:
            self.addButton(button)

    def connect_to(self, connections=None):
        """
        Specify the connections between the signal from this button box and slots specified in a directory

        :param connections: the dictionary linking signals to slots
        """
        if connections is not None:
            for signal, slot in connections.items():
                match signal:
                    case 'accept':
                        self.accepted.connect(slot)
                    case 'reject':
                        self.rejected.connect(slot)
                    case 'click':
                        self.clicked.connect(slot)
                    case 'help':
                        self.helpRequested.connect(slot)


class Dialog(QDialog):
    """
    An extension of QDialog to define forms that may be used to specify plugin options
    """

    def __init__(self, plugin=None, title=None, **kwargs):
        super().__init__(**kwargs)
        self.vert_layout = QVBoxLayout(self)
        self.setLayout(self.vert_layout)
        self.plugin = plugin
        if title is not None:
            self.setWindowTitle(title)
        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Maximum)

    def fit_to_contents(self):
        """
        Adjust the size of the window to fit its contents
        """
        QApplication.processEvents()
        self.adjustSize()

    def addGroupBox(self, title, widget=ParametersFormGroupBox):
        """
        Add a group box to the Dialog window

        :param title: the title of the group box to add
        :param widget: the class of group box
        :return: the group box
        """
        group_box = widget(self)
        group_box.setTitle(title)
        group_box.setStyleSheet(StyleSheets.groupBox)
        return group_box

    def addButtonBox(self, buttons=QDialogButtonBox.Ok | QDialogButtonBox.Close, centered=True):
        """
        Add a button box to the Dialog window

        :param buttons: the buttons to add to the button box
        :param centered: should the buttons be centred
        :return: the button box
        """
        button_box = DialogButtonBox(self, buttons=buttons)
        button_box.setCenterButtons(centered)
        return button_box

    def addButton(self, widget, text=None, icon=None, flat=False):
        """
        Add a button to the Dialog window

        :param widget: the type of button to add
        :param text: the text
        :param icon: the icon
        :param flat: should the button be flat
        :return: the added button
        """
        button = widget(text, icon)
        if flat:
            button.setFlat(True)
        return button

    def arrangeWidgets(self, widget_list):
        """
        Arrange the widgets in the Dialog window vertical layout

        :param widget_list: the list of widgets to add to the vertical layout
        """
        for widget in widget_list:
            self.vert_layout.addWidget(widget)


def set_connections(connections):
    """
    connect a signal to a slot or a list of slots, as defined in a dictionary

    :param connections: the dictionary {signal: slot,...} or {signal: [slot1, slot2,...],...} containing the connections
     to create
    """
    for signal, slot in connections.items():
        if isinstance(slot, list):
            for s in slot:
                signal.connect(s)
        else:
            signal.connect(slot)
