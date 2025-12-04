"""
Classes for handling models that are used to store Parameters data and ensure their synchronization with GUI widgets
"""
import json
from enum import IntEnum
from typing import Any, Generic, TypeVar

import polars
from PySide6.QtCore import QModelIndex, Qt, QStringListModel, Signal, QAbstractTableModel, QPersistentModelIndex
from PySide6.QtGui import QStandardItemModel, QStandardItem

GenericModel = TypeVar('GenericModel')

class StandardItemModel(QStandardItemModel):
    def set_value(self, value: Any) -> None:
        pass

    def value(self) -> Any:
        pass

class ItemModel(StandardItemModel, Generic[GenericModel]):
    """
    Class for all Item models, containing one and only one value of data
    """

    def __init__(self, data: Any = None) -> None:
        super().__init__(1, 1)
        item: QStandardItem = QStandardItem()
        item.setData(data, Qt.EditRole)
        self.setItem(0, 0, item)

    def value(self) -> Any:
        """
        Returns the value of model

        :return: the current value
        """
        return self.data(self.index(0, 0), role=Qt.DisplayRole)

    def set_value(self, value: Any) -> None:
        """
        Sets the value of the model

        :param value: the value
        """
        self.setData(self.index(0, 0), value)

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        """
        Returns the number of rows in the model (always 1)

        :param parent: parent for consistency of signature purposes
        :return: 1
        """
        return 1

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        """
        Returns the number of columns in the model (always 1)

        :param parent: parent for consistency of signature purposes
        :return: 1
        """
        return 1


class StringList(QStringListModel, Generic[GenericModel]):
    """
 Class for string list model
    """

    def __init__(self, data: list[str] = None) -> None:
        super().__init__()
        self._data: list[str] = data if data else []

    @property
    def value(self) -> list[str]:
        """
        Returns the data in the list

        :return: the list of strings in the model
        """
        return self._data

    def data(self, index: QModelIndex, role: IntEnum = Qt.ItemDataRole.DisplayRole) -> str | None:
        """
        Returns the data at the specified index

        :param index: the index for the requested text item
        :param role: the role of data, typically Qt.DisplayRole
        :return: the data at the requested position
        """
        if role == Qt.ItemDataRole.DisplayRole:
            return self._data[index.row()]
        return None

    def rowCount(self, parent: QModelIndex = None) -> int:
        """
        Returns the number of rows in the model

        :param parent: parent argument for signature consistency with the base method
        :return: the number of items
        """
        return len(self._data)

    def add_item(self, item: str) -> None:
        """
        Adds a text item at the end of the current model

        :param item: the text item
        """
        self.beginInsertRows(self.index(len(self._data) - 1, 0), len(self._data), len(self._data))
        self._data.append(item)
        self.endInsertRows()

    def remove_item(self, row: int) -> None:
        """
        Removes from the model the item at the specified row index
        
        :param row: the row to remove
        """
        if 0 <= row < len(self._data):
            self.beginRemoveRows(self.index(row, 0), row, row)
            self._data.pop(row)
            self.endRemoveRows()

    def update_item(self, row: int, item: str) -> None:
        """

        :param row: 
        :param item: 
        """
        if 0 <= row < len(self._data):
            self._data[row] = item
            self.dataChanged.emit(self.index(row, 0), self.index(row, 0), [Qt.ItemDataRole.DisplayRole])


class DictItemModel(StandardItemModel, Generic[GenericModel]):
    """
    Class for Dictionary-based item model. This is used by ChoiceParameter class to hold all choices and the current
    selection
    """
    selection_changed = Signal(int)

    def __init__(self, data_dict: dict[str, Any] = None) -> None:
        super().__init__()
        if data_dict:
            self.set_items(data_dict)
        self.selection: int = 0

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        """
        Returns the number of columns (i.e. always 1 for this model)

        :param parent: parent argument for signature consistency with the base method
        :return: 1
        """
        return 1

    def row(self, index: int) -> QStandardItem:
        """
        Returns the row (item) at position index

        :param index: the index of the row
        :return: the row (item)
        """
        return self.item(index, 0)

    def key(self) -> str | None:
        """
        Returns the currently selected key
        
        :return: the selected key
        """
        try:
            return self.row(self.selection).text()
            # return self.keys()[self.selection]
        except IndexError:
            return None

    def set_value(self, key: str) -> None:
        """
        Sets the current selection to the specified key
        
        :param key: the key to select
        """
        if not isinstance(key, str):
            # key = str(key)
            key = json.dumps(key)
        if key in self.keys():
            self.set_selection(self.keys().index(key))

    def value(self) -> Any:
        """
        Returns the selected value in the model
        
        :return: the select value object
        """
        try:
            # return self.row(self.selection).data(Qt.UserRole)
            selected_value = self.values()[self.selection]
            if selected_value is None:
                return self.key()
            return selected_value
        except IndexError:
            return None

    def rows(self) -> dict[str, Any]:
        """
        Returns all the rows in the model as a dictionary
        
        :return: the rows in the model
        """
        return {self.row(row).text(): self.row(row).data(Qt.ItemDataRole.UserRole) for row in range(self.rowCount())}

    def keys(self) -> list[str]:
        """
        Returns a list of all text keys available in the model
        
        :return: the list of keys
        """
        return [self.row(row).text() for row in range(self.rowCount())]

    def values(self) -> list[Any]:
        """
        Returns a list of all values available in the model
        
        :return: the list of values
        """
        return [self.row(row).data(Qt.ItemDataRole.UserRole) for row in range(self.rowCount())]

    def items(self) -> dict[str, Any]:
        """
        Returns all the available items in the model as a dictionary
        
        :return: the available items dictionary
        """
        return self.rows()

    def set_items(self, data_dict: dict[str, Any]) -> None:
        """
        Sets items for the current data model

        :param data_dict: the dictionary containing the items
        """
        self.clear()
        for key, value in data_dict.items():
            self.add_item({key: value})

    def add_item(self, item: dict[str, Any]) -> None:
        """
        Adds an item, specified by a dictionary, to the current model

        :param item: the dictionary representing the item
        """
        for key, value in item.items():
            if key not in self.keys():
                key_item = QStandardItem(key)
                key_item.setData(value, Qt.ItemDataRole.UserRole)
                self.appendRow([key_item])
            else:
                key_item = self.item(self.keys().index(key))
                key_item.setData(value, Qt.ItemDataRole.UserRole)

    def remove_item(self, row: int) -> None:
        """
        Remove the item at the given position

        :param row: the rank of the row to be removed
        """
        if 0 <= row < self.rowCount():
            self.removeRow(row)

    def set_selection(self, index: int) -> None:
        """
        Set the selected item in the model

        :param index: the index (rank) of the selection
        """
        self.selection = index
        self.selection_changed.emit(index)


class TableModel(QAbstractTableModel):
    """
    A table model based on a polars dataframe. This model can be used to visualize non-editable tabular data
    """

    def __init__(self, data=None):
        super().__init__()
        self.df = data

    def set_data(self, data: polars.DataFrame) -> None:
        """
        Sets the data

        :param data:
        """
        self.df = data

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        """
        Returns the number of rows in the data

        :param parent: the model index parent
        """
        return self.df.shape[0]

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        """
        Returns the number of columns in the data

        :param parent: the model index parent
        """
        return self.df.shape[1]

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = ...) -> str:
        """
        Returns the data for the given role and section in the header with the specified orientation.

        :param section: the section index (row number)
        :param orientation: the orientation
        :param role: the role
        """
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            # return f"Column {section + 1}"
            return self.df.columns[section]
        if orientation == Qt.Vertical and role == Qt.DisplayRole:
            return f"{section + 1}"
        return ''

    def data(self, index: QModelIndex | QPersistentModelIndex, role: int = Qt.DisplayRole) -> object | None:
        """
        Returns the data stored under the given role for the item referred to by the index.

        :param index: the index
        :param role: the role
        """
        column = index.column()
        row = index.row()

        if role in (Qt.DisplayRole, Qt.EditRole):
            return self.df[row, column]
        # elif role == Qt.BackgroundRole:
        #     return QColor(Qt.white)
        # elif role == Qt.TextAlignmentRole:
        #     return Qt.AlignRight
        return None


class EditableTableModel(TableModel):
    """
    A table model based on a polars dataframe. This model can be used to visualize editable tabular data
    """

    def __init__(self, data=None, editable_col=None, editable_row=None):
        super().__init__()
        self.df = data
        if editable_col is None:
            self.editable_col = set()
        else:
            self.editable_col = set(editable_col)
        if editable_row is None:
            self.editable_row = set()
        else:
            self.editable_row = set(editable_row)

    def setData(self, index: QModelIndex | QPersistentModelIndex, value: Any, /, role: int = ...) -> bool:
        """
        Sets the role data for the item at index to value.
        Returns true if successful; otherwise returns false.

        :param index: the index
        :param value: the value to set
        :param role: the role
        """
        column = index.column()
        row = index.row()
        if role == Qt.EditRole:
            self.df[row, column] = value
            self.dataChanged.emit(index, index, [Qt.EditRole, Qt.DisplayRole])
            return True
        return False

    def set_editable_col(self, col: list[int] | int, editable: list[bool] | bool) -> None:
        """
        Sets columns referred to by their index in col list to the editable value

        :param col: the column indices to set enable flag
        :param editable: the editable flags
        """
        for c, e in zip(col, editable):
            if e:
                self.editable_col.add(c)
            else:
                self.editable_col.discard(c)

    def set_editable_row(self, row: list[int] | int, editable: list[bool] | bool) -> None:
        """
        Sets rows referred to by their index in row list to the editable value

        :param row: the row indices to set enable flag
        :param editable: the editable flags
        """
        for r, e in zip(row, editable):
            if e:
                self.editable_row.add(r)
            else:
                self.editable_row.discard(r)

    def flags(self, index: QModelIndex | QPersistentModelIndex) -> Qt.ItemFlag:
        """
        Returns the item flags for the given index.

        :param index: the index
        """
        if not index.isValid():
            return Qt.NoItemFlags
        if self.is_editable(index):
            return Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsEditable
        return Qt.ItemIsSelectable | Qt.ItemIsEnabled

    def is_editable(self, index: QModelIndex | QPersistentModelIndex) -> bool:
        """
        Returns whether the table cell at the index is editable or not

        :param index: the index to check whether it is editable or not
        :return: True if the cell is editable, False otherwise
        """
        if self.editable_row:
            return (index.column() in self.editable_col) and (index.row() in self.editable_row)
        return index.column() in self.editable_col
