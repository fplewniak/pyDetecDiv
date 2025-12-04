"""
Module defining widgets and other utilities for creating windows/forms with a minimum of code
"""
from typing import Any, Self, Type, Union, TypeVar, Callable

from PySide6.QtCore import QStringListModel, QItemSelection, QItemSelectionModel, Signal, Slot, QModelIndex, SignalInstance
from PySide6.QtGui import QIcon, QAction, QContextMenuEvent, QValidator
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QGroupBox, QFormLayout, QLabel, QDialogButtonBox,
                               QSizePolicy, QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox, QAbstractSpinBox, QTableView,
                               QAbstractItemView,
                               QPushButton, QApplication, QRadioButton, QListView, QMenu, QDataWidgetMapper, QWidget, QLayout)

from pydetecdiv.app.models import GenericModel, ItemModel, DictItemModel, StringList
from pydetecdiv import plugins
from pydetecdiv.app.parameters import Parameter


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


StandardButtonCombination = Union["QDialogButtonBox.StandardButton", ...]

GenericGroupBox = TypeVar('GenericGroupBox', bound=QGroupBox)


class GroupBox(QGroupBox):
    """
    an extension of QGroupBox class
    """

    def __init__(self, parent: QWidget, title: str = None) -> None:
        super().__init__(parent)
        if title is not None:
            self.setTitle(title)
        self.setSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Maximum)
        self.layout: QLayout = self.layout()

    @property
    def plugin(self) -> plugins.Plugin | None:
        """
        Property returning the plugin from the top parent of the current widget

        :return: the plugin module or None if it was not found
        """
        parent: QWidget = self.parent()
        while parent:
            if hasattr(parent, 'plugin'):
                return parent.plugin
        return None

    def addSubBox(self, widget: Type[Self], **kwargs: dict[str, Any]) -> Self:
        """
        Adds a sub-box to the current GroupBox

        :param widget: the class of the GroupBox to add as a sub box
        :param kwargs: keywords arguments to pass to the sub box
        :return: the sub box object
        """
        sub_box = widget(self, **kwargs)
        self.layout.addWidget(sub_box)
        return sub_box

    def addOption(self, label: str = None, widget: Type[QWidget] = None, parameter: Parameter = None,
                  enabled: bool = True, **kwargs: dict[str, Any]) -> QWidget:
        """
        add an option to the current Form

        :param enabled: whether this option is enabled
        :param parameter: the Parameter attached to the widget
        :param label: the label for the option
        :param widget: the widget to specify the option value, etc
        :param kwargs: extra args passed to the widget
        :return: the option widget
        """
        ...


class ParametersFormGroupBox(GroupBox):
    """
    an extension of GroupBox class to handle Forms
    """

    def __init__(self, parent: QWidget, title: str = None, show: bool = True) -> None:
        super().__init__(parent, title)
        self.layout: QLayout = QFormLayout(self)
        self.setLayout(self.layout)
        self.setVisible(show)

    def addSubBox(self, widget: Type[GroupBox], **kwargs: dict[str, Any]) -> GroupBox:
        """
        Adds a sub-box to the current ParametersFormGroupBox

        :param widget: the class of the GroupBox to add as a sub box
        :param kwargs: keywords arguments to pass to the sub box
        :return: the sub box object
        """
        sub_box: GroupBox = widget(self, **kwargs)
        self.layout.addRow(sub_box)
        return sub_box

    def addOption(self, label: str = None, widget: Type[QWidget] = None, parameter: Parameter = None,
                  enabled: bool = True, **kwargs: dict[str, Any]) -> QWidget:
        """
        add an option to the current Form

        :param enabled: whether this option is enabled
        :param parameter: the Parameter attached to the widget
        :param label: the label for the option
        :param widget: the widget to specify the option value, etc
        :param kwargs: extra args passed to the widget
        :return: the option widget
        """
        if issubclass(widget, (QPushButton, QDialogButtonBox, QGroupBox)):
            option: QWidget = widget(parent=self, **kwargs)
        else:
            option: QWidget = widget(parent=self, model=parameter.model, **parameter.kwargs(), **kwargs)

        option.setEnabled(enabled)

        if label is None:
            self.layout.addRow(option)
        else:
            self.layout.addRow(QLabel(label), option)
        return option

    def setRowVisible(self, index: int, on: bool = True) -> None:
        """
        set the row defined by index or widget visible or invisible in the form layout

        :param index: the row index or the widget in that row
        :param on: whether the row should be visible or not
        """
        self.layout.setRowVisible(index, on)


class ComboBox(QComboBox):
    """
    an extension of the QComboBox class with a custom model/view architecture
    """

    def __init__(self, parent: QWidget, model: GenericModel = None, editable: bool = False,
                 enabled: bool = True, **kwargs: dict[str, Any]) -> None:
        super().__init__(parent)
        if model is not None and model.rows() is not None:
            self.addItemDict(model.rows())
            self.setModel(model)
            self.setModelColumn(0)
            self.currentIndexChanged.connect(self.model().set_selection)
            self.model().selection_changed.connect(self.setCurrentIndex)
        self.setEditable(editable)
        self.setEnabled(enabled)

    def setCurrentIndex(self, index: int) -> None:
        """
        sets the currently selected index

        :param index: the index to select
        """
        super().setCurrentIndex(index)

    def addItemDict(self, options: dict[str, Any]) -> None:
        """
        add items to the ComboBox as a dictionary

        :param options: dictionary of options specifying labels and corresponding user data {label: userData, ...}
        """
        for label, data in options.items():
            self.addItem(label, userData=data)

    def setItemsDict(self, options: dict[str, Any]) -> None:
        """
        Defines items from a dictionary

        :param options: the dictionary representing the options
        """
        self.clear()
        self.addItemDict(options)

    def setText(self, text: str) -> None:
        """
        sets the current text if text is already an option, otherwise, it adds a new item

        :param text: the text to select or add
        """
        if self.findText(text) != -1:
            self.setCurrentText(text)
        else:
            self.addItem(text)

    def text(self) -> str:
        """
        returns the currently selected text

        :return: the currently selected text
        """
        return self.currentText()

    @property
    def selected(self) -> SignalInstance:
        """
        return property telling whether the current index of this ComboBox has changed

        :return: Signal emitted when the current index has changed (i.e. new selection)
        """
        return self.currentIndexChanged

    @property
    def changed(self) -> Signal:
        """
        return property telling whether the current text of this ComboBox has changed. This overwrites the Pyside
        equivalent method in order to have the same method name for all widgets

        :return: Signal emitted when the current text has changed (i.e. new selection)
        """
        return self.currentTextChanged

    def value(self) -> str | Any:
        """
        method to standardize the way widget values from a form are returned

        :return: the current data if it can be json serialized or the current text of the selected item if it can't
        """
        if self.currentData() is not None:
            return self.currentData()
        return self.currentText()

    def setValue(self, value: str) -> None:
        """
        Defines the currently selected option of the ComboBox

        :param value: the currently selected text
        """
        self.setCurrentText(value)


class ListView(QListView):
    """
    an extension of the QComboBox class
    """

    def __init__(self, parent: QWidget, model: StringList = None, height: int = None, multiselection: bool = False,
                 enabled: bool = True, **kwargs: dict[str, Any]) -> None:
        super().__init__(parent)
        if multiselection:
            self.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        if height is not None:
            self.setFixedHeight(height)
        self.setModel(QStringListModel())
        if model is not None and model.items is not None:
            self.addItemDict(model.items())
        self.setEnabled(enabled)

    def addItemDict(self, options: dict[str, Any]) -> None:
        """
        add items to the ComboBox as a dictionary

        :param options: dictionary of options specifying labels and corresponding user data {label: userData, ...}
        """
        self.items = options
        self.model().setStringList(list(options.keys()))

    @property
    def changed(self) -> SignalInstance:
        """
        return property telling whether the current text of this ComboBox has changed. This overwrites the Pyside
        equivalent method in order to have the same method name for all widgets

        :return: PySide6.QtCore.QModelIndex the new selection model
        """
        return self.selectionModel().currentChanged

    def selection(self) -> list[Any]:
        """
        method to standardize the way widget values from a form are returned

        :return: the current data (if it is defined) or the current text of the selected item
        """
        return [self.items[self.model().data(idx)] for idx in
                sorted(self.selectedIndexes(), key=lambda x: x.row(), reverse=False)]

    def setValue(self):
        """

        """
        # could use setSelectionModel(selectionModel) with selectionModel determined from parameter value
        pass

    def contextMenuEvent(self, e: QContextMenuEvent) -> None:
        """
        Definition of a context menu to clear or toggle selection of sources in list model, remove selected sources from
        the list model, clear the source list model

        :param e: mouse event providing the position of the context menu
        :type e: PySide6.QtGui.QContextMenuEvent
        """
        if self.model().rowCount():
            context = QMenu(self)
            unselect = QAction("Unselect all", self)
            unselect.triggered.connect(self.unselect)
            context.addAction(unselect)
            toggle = QAction("Toggle selection", self)
            toggle.triggered.connect(self.toggle)
            context.addAction(toggle)
            context.addSeparator()
            remove = QAction("Remove selected items", self)
            remove.triggered.connect(self.remove_items)
            context.addAction(remove)
            clear_list = QAction("Clear list", self)
            context.addAction(clear_list)
            clear_list.triggered.connect(self.clear_list)
            context.exec(e.globalPos())

    def unselect(self) -> None:
        """
        Clear selection model
        """
        self.selectionModel().clear()

    def toggle(self) -> None:
        """
        Toggle selection model, selected sources are deselected and unselected ones are selected
        """
        toggle_selection = QItemSelection()
        top_left = self.model().index(0, 0)
        bottom_right = self.model().index(self.model().rowCount() - 1, 0)
        toggle_selection.select(top_left, bottom_right)
        self.selectionModel().select(toggle_selection, QItemSelectionModel.SelectionFlag.Toggle)

    def remove_items(self) -> None:
        """
        Delete selected sources
        """
        for idx in sorted(self.selectedIndexes(), key=lambda x: x.row(), reverse=True):
            self.model().removeRow(idx.row())

    def clear_list(self) -> None:
        """
        Clear the source list
        """
        self.model().removeRows(0, self.model().rowCount())


class ListWidget(QListView):
    """
    An extension of the QListView providing consistency with other custom widgets.
    """

    def __init__(self, parent: QWidget, model: DictItemModel = None, height: int = None, editable: bool = False,
                 multiselection: bool = False, enabled: bool = True,
                 **kwargs: dict[str, Any]) -> None:
        super().__init__(parent)
        # self.setSelectionModel(QItemSelectionModel())
        if multiselection:
            self.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        if model is not None and model.rows() is not None:
            self.setModel(model)
            self.setModelColumn(0)
            self.addItemDict(model.rows())
        self.setEnabled(enabled)
        # self.currentIndexChanged.connect(self.model().set_selection)
        self.selectionModel().currentChanged.connect(self.setCurrentIndex)

    def setCurrentIndex(self, index: QModelIndex) -> None:
        """
        Sets the current index (current selection)

        :param index: the index for the current selection
        """
        self.model().set_selection(index.row())

    def addItemDict(self, options: dict[str, Any]) -> Any:
        """
        add items to the ListWidget as a dictionary

        :param options: dictionary of options specifying labels and corresponding user data {label: userData, ...}
        """
        for text, data in options.items():
            self.addItem(text, userData=data)

    def addItem(self, text: str, userData: Any = None) -> None:
        """
        Adds an item to the list

        :param text: the text to display in the List view
        :param userData: the associated data (can be any type of object)
        """
        self.model().add_item({text: userData})


class LineEdit(QLineEdit):
    """
    an extension of QLineEdit class
    """

    def __init__(self, parent: QWidget, model: ItemModel = None, editable: bool = True, enabled: bool = True,
                 **kwargs: dict[str, Any]) -> None:
        super().__init__(parent)
        self.setEditable(editable)
        self.mapper = QDataWidgetMapper(self)
        self.setModel(model)
        self.setEnabled(enabled)

    @property
    def changed(self):
        """

        :return:
        """
        return self.textChanged

    @property
    def edited(self) -> SignalInstance:
        """
        returns the Signal that editing is finished

        :return: the self.editingFinished signal
        """
        return self.editingFinished

    def setEditable(self, editable: bool = True) -> None:
        """
        Sets the property editable for the LineEdit widget

        :param editable: True if the LineEdit can be edited, False otherwise
        """
        self.setReadOnly(not editable)

    def setModel(self, model: ItemModel) -> None:
        """
        Sets the model for the LineEdit widget

        :param model: the item model containing a str value
        """
        self.mapper.setModel(model)
        self.mapper.addMapping(self, 0, b"text")
        self.mapper.toFirst()


class Label(QLabel):
    """
    an extension of QLabel class
    """

    def __init__(self, parent: QWidget, model: ItemModel = None, **kwargs: dict[str, Any]) -> None:
        super().__init__(parent)
        self.mapper = QDataWidgetMapper(self)
        self.setModel(model)

    # @property
    # def changed(self):
    #     return self.textChanged

    def setModel(self, model: ItemModel) -> None:
        """
        Sets the model for the Label

        :param model: the item model containing a str value
        """
        self.mapper.setModel(model)
        self.mapper.addMapping(self, 0, b"text")
        self.mapper.toFirst()


class PushButton(QPushButton):
    """
    an extension of QPushButton class
    """

    def __init__(self, parent: QWidget, text: str, icon: QIcon = None, flat: bool = False,
                 enabled: bool = True) -> None:
        if icon is None:
            super().__init__(text, parent)
        else:
            super().__init__(icon, text, parent)
        self.setFlat(flat)
        self.setEnabled(enabled)


class AdvancedButton(PushButton):
    """
    an extension of PushButton class to control collapsible group boxes for advanced options
    """

    def __init__(self, parent: QWidget, text: str = 'Advanced options') -> None:
        super().__init__(parent, text=text, icon=QIcon(':icons/show'), flat=True)
        self.group_box: GroupBox | None = None
        self.clicked.connect(self.toggle)

    def hide(self):
        """
        hide the linked group box
        """
        super().hide()
        self.setIcon(QIcon(':icons/show'))
        self.group_box.setVisible(False)

    def linkGroupBox(self, group_box: GroupBox) -> None:
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

    def __init__(self, parent: QWidget, model: ItemModel = None, exclusive: bool = True, enabled: bool = True,
                 **kwargs: dict[str, Any]) -> None:
        super().__init__(parent)
        self.setAutoExclusive(exclusive)
        self.mapper = QDataWidgetMapper(self)
        self.setModel(model)
        self.toggled.connect(self.on_toggled)
        self.setEnabled(enabled)

    def setModel(self, model: ItemModel) -> None:
        """
        Sets the model for the radio button

        :param model: the item model containing a bool value
        """
        if model is not None:
            self.mapper.setModel(model)
            self.mapper.addMapping(self, 0)
            self.mapper.setSubmitPolicy(QDataWidgetMapper.SubmitPolicy.AutoSubmit)
            self.mapper.toFirst()

    @property
    def changed(self) -> SignalInstance:
        """
        return property telling whether the RadioButton value has changed. This overwrites the Pyside equivalent method
         in order to have the same method name for all widgets

        :return: boolean indication whether the value has changed
        """
        return self.toggled

    @Slot(bool)
    def on_toggled(self, checked: bool) -> None:
        """
        Slot setting the value of the bool model when the RadioButton is toggled

        :param checked: bool value indicating whether the button has been checked or unchecked
        """
        self.mapper.model().set_value(checked)


class SpinBox(QSpinBox):
    """
    an extension of the QSpinBox class
    """

    def __init__(self, parent: QWidget, model: ItemModel = None, minimum: int = 1, maximum: int = 4096,
                 single_step: int = 1, adaptive: bool = False, enabled: bool = True, **kwargs: dict[str, Any]) -> None:
        super().__init__(parent)
        self.setRange(minimum, maximum)
        self.setSingleStep(single_step)
        if adaptive:
            self.setStepType(QAbstractSpinBox.StepType.AdaptiveDecimalStepType)
        self.mapper = QDataWidgetMapper(self)
        self.setModel(model)
        self.setEnabled(enabled)

    def setModel(self, model: ItemModel):
        """
        Sets the model for the spin box

        :param model: the item model containing an int value
        """
        if model is not None:
            self.mapper.setModel(model)
            self.mapper.addMapping(self, 0)
            self.mapper.toFirst()

    @property
    def changed(self) -> SignalInstance:
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

    def __init__(self, parent: QWidget, model: ItemModel = None, minimum: float = 0.1, maximum: float = 1.0,
                 decimals: int | None = None, single_step: float = 0.1, adaptive: bool = False, enabled: bool = True,
                 **kwargs: dict[str, Any]) -> None:
        super().__init__(parent)
        self.setRange(minimum, maximum)
        self.setDecimals(15 if decimals is None else decimals)
        self.setSingleStep(single_step)
        if adaptive:
            self.setStepType(QAbstractSpinBox.StepType.AdaptiveDecimalStepType)
        self.mapper: QDataWidgetMapper = QDataWidgetMapper(self)
        self.setModel(model)
        self.setEnabled(enabled)

    def setModel(self, model: ItemModel) -> None:
        """
        Sets the model for the spin box

        :param model: the item model containing a float value
        """
        if model is not None:
            self.mapper.setModel(model)
            self.mapper.addMapping(self, 0)
            self.mapper.toFirst()

    @property
    def changed(self):
        """
        method returning whether the spinbox value has been changed

        :return: boolean, True if the value was changed, False otherwise
        """
        return self.valueChanged

    def validate(self, input_str, pos):
        # If the input is empty, allow it as intermediate
        if not input_str:
            return (QValidator.State.Intermediate,)

        # Check if the input is a valid float or scientific notation
        try:
            float(input_str)
            return (QValidator.State.Acceptable,)
        except ValueError:
            # Check if the input could be a valid scientific notation
            if 'e' in input_str:
                parts = input_str.split('e', 1)
                if len(parts) == 2:
                    mantissa, exponent = parts
                    try:
                        float(mantissa)
                        int(exponent)
                        return (QValidator.State.Acceptable,)
                    except ValueError:
                        pass

        # If the input is not valid but could become valid (e.g., "1e" or "1.23e")
        if input_str.endswith('e') or input_str.endswith('-') or input_str.replace('.', '').replace('-', '').isdigit():
            return (QValidator.State.Intermediate,)

        # If the input is invalid
        return (QValidator.State.Invalid,)

    def valueFromText(self, text):
        try:
            return float(text)
        except ValueError:
            return 0.0

    def textFromValue(self, value):
        # Format the value in scientific notation if needed
        text = "{:.15}".format(value)
        if 'e' in text:
            mantissa, exponent = text.split('e')
            mantissa = mantissa.rstrip('0').rstrip('.') if '.' in mantissa else mantissa
            return f"{mantissa}e{exponent}"
        else:
            return text.rstrip('0').rstrip('.')


class TableView(QTableView):
    """
    an extension of the QTableView widget
    """

    def __init__(self, parent, model=None, enabled=True, **kwargs):
        super().__init__(parent)
        if model is not None:
            self.setModel(model)
        # if model is not None and model.rows() is not None:
        #     self.addItemDict(model.rows())
        #     self.setModel(model)
        #     self.setModelColumn(0)
        #     self.currentIndexChanged.connect(self.model().set_selection)
        #     self.model().selection_changed.connect(self.setCurrentIndex)
        self.setEnabled(enabled)

    # def __init__(self, parent, parameter, multiselection=True, behavior='rows', enabled=True):
    #     super().__init__(parent)
    #     self.model = QSqlQueryModel()
    #     self.setModel(self.model)
    #     if multiselection:
    #         self.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
    #     match behavior:
    #         case 'rows':
    #             self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    #         case _:
    #             pass
    #     self.setEnabled(enabled)
    #
    # def setQuery(self, query):
    #     """
    #     set the SQL query used to feed the model of the table view
    #
    #     :param query: the query
    #     """
    #     self.model.setQuery(query)


class DialogButtonBox(QDialogButtonBox):
    """
    A extension of QDialogButtonBox to add a button box
    """

    def __init__(self, parent: QWidget,
                 buttons: StandardButtonCombination = (QDialogButtonBox.StandardButton.Ok,
                                                       QDialogButtonBox.StandardButton.Close)) -> None:
        super().__init__(parent)
        for button in buttons:
            self.addButton(button)

    def connect_to(self, connections: dict[Signal, Callable] = None) -> None:
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

    def __init__(self, plugin: plugins.Plugin = None, title: str = None, **kwargs: dict[str, Any]) -> None:
        super().__init__(**kwargs)
        self.vert_layout = QVBoxLayout(self)
        self.setLayout(self.vert_layout)
        self.plugin = plugin
        if title is not None:
            self.setWindowTitle(title)
        self.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Maximum)

    def fit_to_contents(self) -> None:
        """
        Adjust the size of the window to fit its contents
        """
        QApplication.processEvents()
        self.adjustSize()

    def addGroupBox(self, title: str = None, widget: Type[GroupBox] = ParametersFormGroupBox) -> GenericGroupBox:
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

    def addButtonBox(self,
                     buttons: StandardButtonCombination = (QDialogButtonBox.StandardButton.Ok,
                                                           QDialogButtonBox.StandardButton.Close),
                     centered: bool = True) -> DialogButtonBox:
        """
        Add a button box to the Dialog window

        :param buttons: the buttons to add to the button box
        :param centered: should the buttons be centred
        :return: the button box
        """
        button_box = DialogButtonBox(self, buttons=buttons)
        button_box.setCenterButtons(centered)
        return button_box

    def addButton(self, widget: Type[QPushButton], text: str = None, icon: QIcon = None,
                  flat: bool = False) -> QPushButton:
        """
        Add a button to the Dialog window

        :param widget: the type of button to add
        :param text: the text
        :param icon: the icon
        :param flat: should the button be flat
        :return: the added button
        """
        button: QPushButton = widget(icon=icon, text=text)
        if flat:
            button.setFlat(True)
        return button

    def arrangeWidgets(self, widget_list: list[QWidget]) -> None:
        """
        Arrange the widgets in the Dialog window vertical layout

        :param widget_list: the list of widgets to add to the vertical layout
        """
        for widget in widget_list:
            self.vert_layout.addWidget(widget)


def set_connections(connections: dict[Signal, Callable]) -> None:
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
