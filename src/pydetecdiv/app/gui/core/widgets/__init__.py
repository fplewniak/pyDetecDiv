"""
Core and absract widgets for application GUI. These widgets provide the basic functionalities for the GUI and are expected
to be extended for concrete or more specific purposes
"""
import os
from collections.abc import Callable
from typing import Any, Type, TypeVar, Union, Self, cast

import polars
from PySide6.QtCore import Signal, Slot, QItemSelectionModel, QItemSelection, SignalInstance
from PySide6.QtGui import QIcon, QAction, QContextMenuEvent, QValidator
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QSizePolicy, QApplication, QDialogButtonBox, QPushButton, QWidget, QGroupBox,
                               QLayout, QLabel, QFormLayout, QTableView, QDataWidgetMapper, QAbstractSpinBox, QDoubleSpinBox,
                               QSpinBox, QRadioButton, QLineEdit, QAbstractItemView, QListView, QMenu, QComboBox, QHBoxLayout,
                               QFileDialog, QHeaderView)
from pydetecdiv.app.parameters import Parameter
from pydetecdiv.app.models import ItemModel, DictItemModel, StringListModel, TableModel, EditableTableModel


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


StandardButtonCombination = Union["QDialogButtonBox.StandardButton"]

GenericGroupBox = TypeVar('GenericGroupBox', bound=QGroupBox)


def paramwidget_args(parameter: Parameter, param_args: dict[str, Any] | None) -> dict[str, Any]:
    """
    Return the arguments defined in param_args (declared when creating the Dialog object) and corresponding to the parameter when
    adding an option to a group box. It is thus possible to specify GUI-specific arguments during GUI creation without having to
    define them initially when creating parameters.

    :param parameter: the parameter
    :param param_args: the GUI-specific arguments
    :return: the parameter's arguments
    """
    if param_args is not None and parameter.name in param_args:
        multiple_val = [arg for arg in param_args[parameter.name] if arg in parameter.kwargs()]
        if multiple_val:
            param_args[parameter.name].pop(*multiple_val)
        # for arg in param_args[parameter.name]:
        #     if arg in parameter.kwargs():
        #         # parameter.__dict__.pop(arg)
        #         param_args[parameter.name].pop(arg)
        return param_args[parameter.name]
    return {}


class GroupBox(QGroupBox):
    """
    an extension of QGroupBox class
    """

    def __init__(self, parent: QWidget, title: str | None = None, show: bool = True, **kwargs: dict[str, Any]) -> None:
        super().__init__(parent, **kwargs)
        if title is not None:
            self.setTitle(title)
        self.setSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Maximum)
        self.layout: QLayout = cast(QLayout, self.layout())
        self.setVisible(show)
        self.parameter_widgets = {}
        self.option = {}

    def parameter_widget_factory(self, parameter: Parameter, **kwargs) -> QWidget:
        """
        A factory method to create a parameter widget for a given parameter
        :param parent: the parent for the created parameter widget
        :param parameter: the parameter
        :param kwargs: any additional keyword arguments passed to the created widget
        :return:
        """
        parameter_widgets = {
            'IntParameter'       : SpinBox,
            'FloatParameter'     : DoubleSpinBox,
            'StringParameter'    : LineEdit,
            'CheckParameter'     : RadioButton,
            'ChoiceParameter'    : ComboBox,
            'DirParameter'       : DirChooser,
            'FileParameter'      : FileChooser,
            'StringListParameter': ListView,
            }
        self.parameter_widgets[parameter.name] = parameter_widgets[parameter.type](parent=self, **parameter.kwargs(), **kwargs)
        return self.parameter_widgets[parameter.name]

    def addSubBox(self, widget: Type[Self], expandable: bool = False, show: bool = True, title: str | None = None,
                  parameters: list | None = None, widget_args: dict[str, Any] | None = None,
                  **kwargs: dict[str, Any]) -> 'Self | ExpandCollapseButton':
        """
        Adds a sub-box to the current GroupBox

        :param expandable: True if sub-box can be collapsed/expanded
        :param show: True if expandable sub-box should be shown (expanded) by default
        :param title: the title of the sub-box (optional)
        :param parameters: the list of parameters
        :param widget_args: a dictionary of keyword arguments passed to the created widget (optional)
        :param widget: the class of the GroupBox to add as a sub box
        :param kwargs: keywords arguments to pass to the sub box
        :return: the sub box object
        """
        if expandable:
            sub_box: ExpandCollapseButton = ExpandCollapseButton(self, text=title, show=show)
            sub_box.linkGroupBox(widget(self, **kwargs))
        else:
            sub_box: 'Self | ExpandCollapseButton' = widget(self, title=title, **kwargs)
            self.layout.addWidget(sub_box)
        if parameters is not None:
            for parameter in parameters:
                sub_box.addOption(parameter, **paramwidget_args(parameter, widget_args))
        return sub_box

    def addOption(self, parameter: Parameter | None = None, label: bool = True, widget: Type[QWidget] | None = None,
                  **kwargs: dict[str, Any]) -> QWidget:
        """
        add an option to the current Form

        :param enabled: whether this option is enabled
        :param parameter: the Parameter attached to the widget
        :param label: the label for the option
        :param widget: the widget to specify the option value, etc
        :param kwargs: extra args passed to the widget
        :return: the option widget
        """

    def addWidget(self, widget: Type[QWidget], **kwargs: dict[str, Any]) -> Type[QWidget]:
        """
        Method to add a widget to the Group box. This method should be implemented by subclasses
        """

    def __getattr__(self, item: str) -> QWidget:
        """
        Dunder method to allow access to option widget using attribute syntax
        :param item: the name of the option (the same as the corresponding parameter)
        :return: the option widget
        """
        return self.option[item]


class InfoGroupBox(GroupBox):
    """
    an extension of GroupBox class to show information
    """

    def __init__(self, parent: QWidget, title: str | None = None, show: bool = True, layout: QLayout | None = None) -> None:
        super().__init__(parent, title)
        self.layout: QLayout = QVBoxLayout(self) if layout is None else layout
        self.setLayout(self.layout)
        self.setVisible(show)

    def addWidget(self, widget: Type[QWidget], **kwargs: dict[str, Any]) -> Type[QWidget]:
        self.layout.addWidget(widget(**kwargs))
        return widget


class ParametersFormGroupBox(GroupBox):
    """
    an extension of GroupBox class to handle Forms
    """

    def __init__(self, parent: QWidget, title: str | None = None, show: bool = True, **kwargs: dict[str, Any]) -> None:
        super().__init__(parent, title, **kwargs)
        self.layout: QFormLayout = QFormLayout(self)
        self.setLayout(self.layout)
        self.setVisible(show)

    def addSubBox(self, widget: Type[GroupBox], expandable: bool = False, show: bool = True, title: str | None = None,
                  parameters: list | None = None, widget_args: dict[str, Any] | None = None,
                  **kwargs: dict[str, Any]) -> 'GroupBox | ExpandCollapseButton':
        """
        Adds a sub-box to the current ParametersFormGroupBox

        :param expandable: True if sub-box can be collapsed/expanded
        :param show: True if expandable sub-box should be shown (expanded) by default
        :param title: the title of the sub-box (optional)
        :param parameters: the list of parameters
        :param widget_args: a dictionary of keyword arguments passed to the created widget (optional)
        :param widget: the class of the GroupBox to add as a sub box
        :param kwargs: keywords arguments to pass to the sub box
        :return: the sub box object
        """
        if expandable:
            sub_box: ExpandCollapseButton = ExpandCollapseButton(self, text=title, show=show)
            sub_box.linkGroupBox(widget(self, **kwargs))
        else:
            sub_box: GroupBox = widget(self, title=title, **kwargs)
            self.layout.addRow(sub_box)
        if parameters is not None:
            for parameter in parameters:
                sub_box.addOption(parameter, **paramwidget_args(parameter, widget_args))
        return sub_box

    def addOption(self, parameter: Parameter | None = None, label: bool = True, widget: Type[QWidget] | None = None,
                  **kwargs: dict[str, Any]) -> QWidget:
        """
        add an option to the current Form

        :param enabled: whether this option is enabled
        :param parameter: the Parameter attached to the widget
        :param label: the label for the option
        :param widget: the widget to specify the option value, etc
        :param kwargs: extra args passed to the widget
        :return: the option widget
        """
        # if widget is specified, it takes precedence over the standard widget normally used for the specified parameter. It is
        # the responsibility of the developer though to ensure the widget can manage the said parameter and the associated model
        parameter = cast(Parameter, parameter)
        if widget is not None:
            self.option[parameter.name]: QWidget = widget(parent=self, **parameter.kwargs(), **kwargs)
        else:
            self.option[parameter.name]: QWidget = self.parameter_widget_factory(parameter, **kwargs)

        if not label:
            self.layout.addRow(self.option[parameter.name])
        else:
            self.layout.addRow(QLabel(parameter.label), self.option[parameter.name])
        # parameter.should_be_saved = True
        return self.option[parameter.name]

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

    def __init__(self, parent: QWidget, qmodel: DictItemModel = DictItemModel(), editable: bool = False,
                 enabled: bool = True, **kwargs: dict[str, Any]) -> None:
        super().__init__(parent)
        if qmodel is not None and qmodel.rows() is not None:
            self.addItemDict(qmodel.rows())
            self.setModel(qmodel)
            self.qmodel = qmodel
            self.setModelColumn(0)
            # self.currentIndexChanged.connect(self.qmodel.set_selection)
            # self.qmodel.selection_changed.connect(self.setCurrentIndex)
            # self.setCurrentIndex(self.qmodel.selection)
        self.setEditable(editable)
        self.setEnabled(enabled)

        self.view().setSelectionModel(qmodel.selection_model)
        self.qmodel.selection_model.selectionChanged.connect(self._on_selection_changed)

    def _on_selection_changed(self,selected, deselected):
        """Update the combo box index when the model's selection changes."""
        if selected.indexes():
            self.setCurrentIndex(selected.indexes()[0].row())

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
    def changed(self) -> SignalInstance:
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
    an extension of the QListView class providing generic methods for managing lists
    """

    def __init__(self, parent: QWidget, qmodel: StringListModel = StringListModel(), height: int | None = None,
                 multiselection: bool = False, enabled: bool = True, **kwargs: dict[str, Any]) -> None:
        super().__init__(parent)
        self.multiselection = multiselection
        if multiselection:
            self.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        if height is not None:
            self.setFixedHeight(height)
        if qmodel is not None and qmodel.items() is not None:
            self.qmodel: StringListModel = qmodel
        else:
            self.qmodel: StringListModel = StringListModel()
        self.setModel(self.qmodel)
        self.setEnabled(enabled)

    def addItems(self, items: list[str]) -> None:
        """
        add items to the ComboBox as a dictionary

        :param options: dictionary of options specifying labels and corresponding user data {label: userData, ...}
        """
        self.qmodel.add_items(items)

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
        return [self.qmodel.items()[idx.row()] for idx in
                sorted(self.selectionModel().selectedRows(), key=lambda x: x.row(), reverse=False)]

    def select_value(self):
        """
        Set the List view content, should be implemented by subclasses
        """
        # could use setSelectionModel(selectionModel) with selectionModel determined from parameter value

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
            if self.multiselection:
                toggle = QAction("Toggle selection", self)
                toggle.triggered.connect(self.toggle)
                context.addAction(toggle)
            context.addSeparator()
            add = QAction("Add item", self)
            add.triggered.connect(self.add_item)
            context.addAction(add)
            remove = QAction("Remove selected items", self)
            remove.triggered.connect(self.remove_items)
            context.addAction(remove)
            clear_list = QAction("Clear list", self)
            context.addAction(clear_list)
            clear_list.triggered.connect(self.clear_list)
            # test_selection = QAction("Test selection", self)
            # context.addAction(test_selection)
            # test_selection.triggered.connect(lambda : print(self.selection()))
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

    def add_item(self) -> None:
        """
        Add an item to the list
        """
        self.qmodel.add_item('new')

    def remove_items(self) -> None:
        """
        Delete selected sources
        """
        for idx in sorted(self.selectedIndexes(), key=lambda x: x.row(), reverse=True):
            self.qmodel.remove_item(idx.row())

    def clear_list(self) -> None:
        """
        Clear the source list
        """
        self.qmodel.clear()


class DictListView(QListView):
    """
    An extension of ListView for dictionaries: the key (str) is displayed on the view, and the value is the corresponding data. This
    allows to use a ListView to manage and select any kind of object that has a name.
    """
    def __init__(self, parent: QWidget, qmodel: DictItemModel = DictItemModel(), height: int | None = None,
                 multiselection: bool = False, enabled: bool = True, **kwargs: dict[str, Any]) -> None:
        super().__init__(parent)
        if height is not None:
            self.setFixedHeight(height)
        if qmodel is not None and qmodel.items() is not None:
            self.qmodel: DictItemModel = qmodel
        else:
            self.qmodel: DictItemModel = DictItemModel()
        self.setModel(qmodel)
        self.setSelectionModel(qmodel.selection_model)
        if multiselection:
            self.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        self.setModelColumn(0)
        self.setEnabled(enabled)

    def selection(self) -> list[Any]:
        """
        method to standardize the way widget values from a form are returned

        :return: the current data (if it is defined) or the current text of the selected item
        """
        return [self.qmodel.values()[idx.row()] for idx in self.selectionModel().selectedRows()]


# class ListWidget(QListView):
#     """
#     An extension of the QListView providing consistency with other custom widgets.
#     """
#
#     def __init__(self, parent: QWidget, qmodel: DictItemModel = DictItemModel(), height: int | None = None, editable: bool = False,
#                  multiselection: bool = False, enabled: bool = True,
#                  **kwargs: dict[str, Any]) -> None:
#         super().__init__(parent)
#         # self.setSelectionModel(QItemSelectionModel())
#         if multiselection:
#             self.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
#         if qmodel is not None and qmodel.rows() is not None:
#             self.setModel(qmodel)
#             self.setModelColumn(0)
#             self.addItemDict(qmodel.rows())
#         self.setEnabled(enabled)
#         # self.currentIndexChanged.connect(self.model().set_selection)
#         self.selectionModel().currentChanged.connect(self.setCurrentIndex)
#
#     def setCurrentIndex(self, index: QModelIndex) -> None:
#         """
#         Sets the current index (current selection)
#
#         :param index: the index for the current selection
#         """
#         self.model().set_selection(index.row())
#
#     def addItemDict(self, options: dict[str, Any]) -> Any:
#         """
#         add items to the ListWidget as a dictionary
#
#         :param options: dictionary of options specifying labels and corresponding user data {label: userData, ...}
#         """
#         self.items = options
#         for text, data in options.items():
#             self.addItem(text, userData=data)
#
#     def addItem(self, text: str, userData: Any = None) -> None:
#         """
#         Adds an item to the list
#
#         :param text: the text to display in the List view
#         :param userData: the associated data (can be any type of object)
#         """
#         self.model().add_item({text: userData})
#
#     def selection(self) -> list[Any]:
#         """
#         method to standardize the way widget values from a form are returned
#
#         :return: the current data (if it is defined) or the current text of the selected item
#         """
#         return [self.items[self.model().data(idx)] for idx in
#                 sorted(self.selectedIndexes(), key=lambda x: x.row(), reverse=False)]


class LineEdit(QLineEdit):
    """
    an extension of QLineEdit class
    """

    def __init__(self, parent: QWidget, qmodel: ItemModel = ItemModel(), editable: bool = True, enabled: bool = True,
                 **kwargs: dict[str, Any]) -> None:
        super().__init__(parent)
        self.setEditable(editable)
        self.mapper = QDataWidgetMapper(self)
        self.setModel(qmodel)
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

    def setText(self, arg__1, /):
        """
        set text of line text editor
        :param arg__1: the text
        """
        super().setText(arg__1)
        self.editingFinished.emit()

    def setEditable(self, editable: bool = True) -> None:
        """
        Sets the property editable for the LineEdit widget

        :param editable: True if the LineEdit can be edited, False otherwise
        """
        self.setReadOnly(not editable)

    def setModel(self, qmodel: ItemModel) -> None:
        """
        Sets the model for the LineEdit widget

        :param model: the item model containing a str value
        """
        self.mapper.setModel(qmodel)
        self.mapper.addMapping(self, 0, b"text")
        self.mapper.toFirst()
        # self.changed.connect(lambda: self.mapper.submit())
        self.editingFinished.connect(lambda: self.mapper.submit())


class PathChooser(QWidget):
    """
    A generic class providing the basic methods for file and directory choosers
    """
    def __init__(self, parent: QWidget, qmodel: ItemModel = ItemModel(), editable: bool = True, enabled: bool = True,
                 min_width=350, **kwargs: dict[str, Any]) -> None:
        super().__init__(parent)
        layout = QHBoxLayout()
        self.path: LineEdit = LineEdit(self, qmodel=qmodel, editable=editable, enabled=enabled)
        self.path.setMinimumWidth(min_width)
        button_path = QPushButton(self)
        button_path.setIcon(QIcon(":icons/file_chooser"))
        layout.addWidget(self.path)
        layout.addWidget(button_path)
        self.setLayout(layout)
        self.setEnabled(enabled)
        button_path.clicked.connect(self.select_path)

    def select_path(self) -> None:
        """
        Select a path
        """


class FileChooser(PathChooser):
    """
    A widget  to choose a file
    """
    def __init__(self, parent: QWidget, qmodel: ItemModel = ItemModel(), editable: bool = True, require_existing: bool = False,
                 enabled: bool = True, min_width=350, filters: list[str] | None = None, selected_filter: int = 0,
                 **kwargs: dict[str, Any]) -> None:
        super().__init__(parent, qmodel, editable, enabled, min_width, **kwargs)

        if filters is not None:
            self.filters = filters
        else:
            self.filters = ['All files (*)']
        self.selected_filter = selected_filter if selected_filter < len(self.filters) else 0
        self.require_existing = require_existing

    def select_path(self) -> None:
        """
        Select a file path
        """
        self.select_file()

    def select_file(self) -> None:
        """
        Select a file
        """
        current_dir = os.path.dirname(self.path.text())
        if self.require_existing:
            file_name, _ = QFileDialog.getOpenFileName(self, caption='Choose file', dir=current_dir, filter=";;".join(self.filters),
                                                       selectedFilter=self.filters[self.selected_filter])
        else:
            file_name, _ = QFileDialog.getSaveFileName(self, caption='Choose file', dir=current_dir, filter=";;".join(self.filters),
                                                       selectedFilter=self.filters[self.selected_filter])
        if file_name:
            self.path.setText(file_name)


class DirChooser(PathChooser):
    """
    A widget to choose a directory
    """
    def __init__(self, parent: QWidget, qmodel: ItemModel = ItemModel(), editable: bool = True,
                 enabled: bool = True, min_width=350, **kwargs: dict[str, Any]) -> None:
        super().__init__(parent, qmodel, editable, enabled, min_width, **kwargs)

    def select_path(self) -> None:
        """
        Select a directory
        """
        self.select_dir()

    def select_dir(self) -> None:
        """
        select a directory
        """
        current_dir = self.path.text()
        path = QFileDialog.getExistingDirectory(self, caption='Choose directory', dir=current_dir)
        if path:
            self.path.setText(path)


class Label(QLabel):
    """
    an extension of QLabel class
    """

    def __init__(self, parent: QWidget, qmodel: ItemModel = ItemModel(), **kwargs: dict[str, Any]) -> None:
        super().__init__(parent)
        self.mapper = QDataWidgetMapper(self)
        self.setModel(qmodel)

    # @property
    # def changed(self):
    #     return self.textChanged

    def setModel(self, qmodel: ItemModel) -> None:
        """
        Sets the model for the Label

        :param qmodel: the item model containing a str value
        """
        self.mapper.setModel(qmodel)
        self.mapper.addMapping(self, 0, b"text")
        self.mapper.toFirst()


class PushButton(QPushButton):
    """
    an extension of QPushButton class
    """

    def __init__(self, parent: QWidget, text: str, icon: QIcon | None = None, flat: bool = False,
                 enabled: bool = True) -> None:
        if icon is None:
            super().__init__(text, parent)
        else:
            super().__init__(icon, text, parent)
        self.setFlat(flat)
        self.setEnabled(enabled)


class ExpandCollapseButton(PushButton):
    """
    an extension of PushButton class to control collapsible group boxes
    """

    def __init__(self, parent: QWidget, text: str | None = None, show: bool = True) -> None:
        super().__init__(parent, text=text, icon=QIcon(':icons/show'), flat=True)
        self.group_box: GroupBox | None = None
        self.clicked.connect(self.toggle)
        self.show = show

    def hide(self):
        """
        hide the linked group box
        """
        self.setIcon(QIcon(':icons/show'))
        self.group_box.setVisible(False)

    def linkGroupBox(self, group_box: GroupBox) -> None:
        """
        link this advanced button to a group box whose expansion or collapse should be controlled by this button

        :param group_box: the group box to link to this button
        """
        self.group_box = group_box
        if isinstance(self.parent().layout, QFormLayout):
            self.parent().layout.addRow(self)
            self.parent().layout.addRow(self.group_box)
        else:
            self.parent().layout.addWidget(self)
            self.parent().layout.addWidget(self.group_box)
        if self.show:
            self.setIcon(QIcon(':icons/hide'))
        self.group_box.setVisible(self.show)

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

    def addSubBox(self, widget: Type[Self], expandable: bool = False, show: bool = True, title: str | None = None,
                  parameters: list | None = None, widget_args: dict[str, Any] | None = None,
                  **kwargs: dict[str, Any]) -> Self:
        """
        Add a sub box to the current collapsable group box
        :param widget: the widget to add
        :param expandable: whether the widget is expandable or not
        :param show: whether the widget is shown or not
        :param title: the sub box title
        :param kwargs: additional keyword arguments
        """
        return self.group_box.addSubBox(widget, expandable, show, title, parameters, widget_args, **kwargs)

    def addOption(self, parameter: Parameter | None = None, label: bool = True, widget: Type[QWidget] | None = None,
                  **kwargs: dict[str, Any]) -> QWidget:
        """
        Add an option to the current collapsable group box
        :param parameter: the parameter to add
        :param label: the label to show
        :param enabled: whether the option is enabled
        :param widget: the widget to show
        :param kwargs: additional keyword arguments
        """
        return self.group_box.addOption(parameter, label, widget, **kwargs)


class RadioButton(QRadioButton):
    """
    an extension of the QRadioButton class
    """

    def __init__(self, parent: QWidget, qmodel: ItemModel = ItemModel(), exclusive: bool = True, enabled: bool = True,
                 **kwargs: dict[str, Any]) -> None:
        super().__init__(parent)
        self.setAutoExclusive(exclusive)
        self.mapper = QDataWidgetMapper(self)
        self.setModel(qmodel)
        self.toggled.connect(self.on_toggled)
        self.setEnabled(enabled)

    def setModel(self, qmodel: ItemModel) -> None:
        """
        Sets the model for the radio button

        :param model: the item model containing a bool value
        """
        if qmodel is not None:
            self.mapper.setModel(qmodel)
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

    def __init__(self, parent: QWidget, qmodel: ItemModel = ItemModel(), minimum: int = 1, maximum: int = 4096,
                 single_step: int = 1, adaptive: bool = False, enabled: bool = True, **kwargs: dict[str, Any]) -> None:
        super().__init__(parent)
        self.setRange(minimum, maximum)
        self.setSingleStep(single_step)
        if adaptive:
            self.setStepType(QAbstractSpinBox.StepType.AdaptiveDecimalStepType)
        self.mapper = QDataWidgetMapper(self)
        self.setModel(qmodel)
        self.setEnabled(enabled)

    def setModel(self, qmodel: ItemModel):
        """
        Sets the model for the spin box

        :param model: the item model containing an int value
        """
        if qmodel is not None:
            self.mapper.setModel(qmodel)
            self.mapper.addMapping(self, 0)
            self.mapper.toFirst()
            self.changed.connect(lambda _: self.mapper.submit())

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

    def __init__(self, parent: QWidget, qmodel: ItemModel = ItemModel(), minimum: float = 0.1, maximum: float = 1.0,
                 decimals: int = 15, single_step: float = 0.1, adaptive: bool = False, enabled: bool = True,
                 **kwargs: dict[str, Any]) -> None:
        super().__init__(parent)
        self.setRange(minimum, maximum)
        self.setDecimals(decimals)
        self.setSingleStep(single_step)
        if adaptive:
            self.setStepType(QAbstractSpinBox.StepType.AdaptiveDecimalStepType)
        self.mapper: QDataWidgetMapper = QDataWidgetMapper(self)
        self.setModel(qmodel)
        self.setEnabled(enabled)

    def setModel(self, qmodel: ItemModel) -> None:
        """
        Sets the model for the spin box

        :param qmodel: the item model containing a float value
        """
        if qmodel is not None:
            self.mapper.setModel(qmodel)
            self.mapper.addMapping(self, 0)
            self.mapper.toFirst()
            # self.changed.connect(lambda _: self.mapper.submit())

    @property
    def changed(self) -> SignalInstance:
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

    def __init__(self, parent, qmodel: TableModel = TableModel(polars.DataFrame()), enabled=True, **kwargs):
        super().__init__(parent)
        if qmodel is not None:
            self.setModel(qmodel)
            self._model: TableModel = qmodel
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.horizontalHeader().setStretchLastSection(True)
        self.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.verticalHeader().setVisible(False)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        # self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        # if model is not None and model.rows() is not None:
        #     self.addItemDict(model.rows())
        #     self.setModel(model)
        #     self.setModelColumn(0)
        #     self.currentIndexChanged.connect(self.model().set_selection)
        #     self.model().selection_changed.connect(self.setCurrentIndex)
        self.setEnabled(enabled)

    def setModel(self, model: TableModel, /):
        super().setModel(model)
        self._model = model

    def row_counts(self):
        return self._model.rowCount()

    def is_empty(self):
        return self.row_counts() == 0 or self._model.df[0][0] is None

    def selected_rows(self, data=False):
        selected_rows_idx = [selection.row() for selection in self.selectionModel().selectedRows()]
        if data:
            return self._model.df.gather(selected_rows_idx)
        return selected_rows_idx

    @property
    def data(self):
        return self._model.df

    def set_data(self, data):
        self._model.set_data(data)


class EditableTableView(TableView):
    def __init__(self, parent, qmodel: EditableTableModel = EditableTableModel(polars.DataFrame()), enabled=True, **kwargs):
        super().__init__(parent, qmodel, enabled, **kwargs)
        if qmodel is not None:
            self.setModel(qmodel)
            self._model: EditableTableModel = qmodel

    def add_rows(self, df: polars.DataFrame) -> None:
        self._model.add_rows(df)

    def delete_row(self, row_id):
        self._model.delete_row(row_id)


class DialogButtonBox(QDialogButtonBox):
    """
    A extension of QDialogButtonBox to add a button box
    """

    def __init__(self, parent: QWidget,
                 buttons: StandardButtonCombination = QDialogButtonBox.StandardButton.Ok |
                                                       QDialogButtonBox.StandardButton.Close) -> None:
        super().__init__(parent)
        for button in buttons:
            self.addButton(button)

    def connect_to(self, connections: dict[Signal, Callable] | None = None) -> None:
        """
        Specify the connections between the signal from this button box and slots specified in a dictionary

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

    def __init__(self, title: str | None = None, **kwargs: dict[str, Any]) -> None:
        super().__init__(**kwargs)
        self.vert_layout = QVBoxLayout(self)
        self.setLayout(self.vert_layout)
        if title is not None:
            self.setWindowTitle(title)
        self.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Maximum)

    def fit_to_contents(self) -> None:
        """
        Adjust the size of the window to fit its contents
        """
        QApplication.processEvents()
        self.adjustSize()

    def addGroupBox(self, title: str | None = None, widget: Type[GroupBox] = ParametersFormGroupBox, parameters: list | None = None,
                    widget_args: dict[str, Any] | None = None, expandable: bool = False, show: bool = True,
                    **kwargs: dict[str, Any]) -> GroupBox:
        """
        Add a group box to the Dialog window

        :param widget_args: extra widget parameters
        :param show: When expandable is True, the group box is expanded if show is True, otherwise the group box is collapsed. When
         expandable is False, this argument has no effect.
        :param expandable: If True, the group box is expandable/collapsable, otherwise the group box is always visible
        :param parameters: the list of parameters in the group box
        :param title: the title of the group box to add
        :param widget: the class of group box
        :return: the group box
        """
        group_box = widget(self)
        group_box.setTitle(cast(str, title))
        group_box.setStyleSheet(StyleSheets.groupBox)
        if expandable:
            group_box.addSubBox(widget=widget, expandable=expandable, show=show, parameters=parameters, widget_args=widget_args,
                                **kwargs)
        else:
            if parameters is not None:
                for parameter in parameters:
                    group_box.addOption(parameter, **paramwidget_args(parameter, widget_args))
            else:
                # group_box.addSubBox(widget=widget, expandable=expandable, show=show, **kwargs)
                pass
        return group_box

    def addButtonBox(self,
                     buttons: StandardButtonCombination = QDialogButtonBox.StandardButton.Ok
                                                          | QDialogButtonBox.StandardButton.Close,
                     centered: bool = True) -> DialogButtonBox:
        """
        Add a button box to the Dialog window

        :param buttons: the buttons to add to the button box
        :param centered: should the buttons be centred
        :return: the button box
        """
        button_box = DialogButtonBox(self, buttons=buttons)
        button_box.setCenterButtons(centered)
        button_box.rejected.connect(self.close)
        return button_box

    def addButton(self, widget: Type[QPushButton], text: str | None = None, icon: QIcon | None = None,
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


def set_connections(connections: dict[SignalInstance, Callable | list[Callable]]) -> None:
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


# def connect_enabling_signals(action: QAction | QMenu, signals: list[SignalInstance] | SignalInstance) -> None:
#     if signals is not None:
#         if not isinstance(signals, list):
#             signals = [signals]
#         for signal in signals:
#             signal.connect(lambda: action.setEnabled(True))
#     else:
#         action.setEnabled(action.enabled_default)
