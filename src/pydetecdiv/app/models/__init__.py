"""
Classes for handling models that are used to store Parameters data and ensure their synchronization with GUI widgets
"""
import json
from enum import IntEnum
from typing import Any

from PySide6.QtCore import QModelIndex, Qt, QStringListModel, Signal
from PySide6.QtGui import QStandardItemModel, QStandardItem


class ItemModel(QStandardItemModel):
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


class StringList(QStringListModel):
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

    def data(self, index: QModelIndex, role: IntEnum = Qt.DisplayRole) -> str | None:
        """
        Returns the data at the specified index

        :param index: the index for the requested text item
        :param role: the role of data, typically Qt.DisplayRole
        :return: the data at the requested position
        """
        if role == Qt.DisplayRole:
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

    def update_item(self, row: int, item: object) -> object:
        """

        :param row: 
        :param item: 
        """
        if 0 <= row < len(self._data):
            self._data[row] = item
            self.dataChanged.emit(self.index(row, 0), self.index(row, 0), [Qt.DisplayRole])


class DictItemModel(QStandardItemModel):
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
        return {self.row(row).text(): self.row(row).data(Qt.UserRole) for row in range(self.rowCount())}

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
        return [self.row(row).data(Qt.UserRole) for row in range(self.rowCount())]

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
                key_item.setData(value, Qt.UserRole)
                self.appendRow([key_item])
            else:
                key_item = self.item(self.keys().index(key))
                key_item.setData(value, Qt.UserRole)

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
