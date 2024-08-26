from PySide6.QtCore import QAbstractItemModel, QModelIndex, Qt, QAbstractListModel, QStringListModel, \
    QAbstractTableModel, Signal
from PySide6.QtGui import QStandardItemModel, QStandardItem


class ItemModel(QAbstractItemModel):
    def __init__(self, data: object = None, parent=None):
        super().__init__(parent)
        self._data = data

    def value(self):
        return self.data(self.index(0, 0))

    def set_value(self, value):
        self.setData(self.index(0,0), value)

    def rowCount(self, parent=QModelIndex()):
        return 1

    def columnCount(self, parent=QModelIndex()):
        return 1

    def data(self, index, role=Qt.DisplayRole):
        if role == Qt.DisplayRole or role == Qt.EditRole:
            return self._data
        return None

    def setData(self, index, value, role=Qt.EditRole):
        if role == Qt.EditRole:
            self._data = value
            self.dataChanged.emit(index, index, [role])
            return True
        return False

    def index(self, row, column, parent=QModelIndex()):
        if not parent.isValid() and row == 0 and column == 0:
            return self.createIndex(row, column)
        return QModelIndex()

    def parent(self, index):
        return QModelIndex()


class StringList(QStringListModel):
    def __init__(self, data=None, parent=None):
        super().__init__(parent)
        self._data = data if data else []

    @property
    def value(self):
        return self._data

    def data(self, index, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            return self._data[index.row()]
        return None

    def rowCount(self, parent=None):
        return len(self._data)

    def addItem(self, item):
        # Commence l'insertion à la fin de la liste
        self.beginInsertRows(self.index(len(self._data) - 1, 0), len(self._data), len(self._data))
        self._data.append(item)
        self.endInsertRows()

    def removeItem(self, row):
        if 0 <= row < len(self._data):
            # Commence la suppression de la ligne spécifiée
            self.beginRemoveRows(self.index(row, 0), row, row)
            self._data.pop(row)
            self.endRemoveRows()

    def updateItem(self, row, item):
        if 0 <= row < len(self._data):
            self._data[row] = item
            # Émet un signal que la donnée a été changée
            self.dataChanged.emit(self.index(row, 0), self.index(row, 0), [Qt.DisplayRole])


class DictItemModel(QStandardItemModel):
    selection_changed = Signal(int)

    def __init__(self, data_dict=None, parent=None):
        super().__init__(parent)
        if data_dict:
            self.set_items(data_dict)
        self.selection = 0

    def columnCount(self, parent=QModelIndex()):
        return 1

    def row(self, index):
        return self.item(index, 0)

    def key(self):
        try:
            return self.row(self.selection).text()
            # return self.keys()[self.selection]
        except IndexError:
            return None

    def set_value(self, key):
        self.set_selection(self.keys().index(key))

    def value(self):
        try:
            # return self.row(self.selection).data(Qt.UserRole)
            selected_value = self.values()[self.selection]
            if selected_value is None:
                return self.key()
            return selected_value
        except IndexError:
            return None

    def rows(self):
        return {self.row(row).text(): self.row(row).data(Qt.UserRole) for row in range(self.rowCount())}

    def keys(self):
        return [self.row(row).text() for row in range(self.rowCount())]

    def values(self):
        return [self.row(row).data(Qt.UserRole) for row in range(self.rowCount())]

    def items(self):
        return self.rows()

    def set_items(self, data_dict):
        self.clear()
        for key, value in data_dict.items():
            self.add_item({key: value})

    def add_item(self, item):
        for key, value in item.items():
            key_item = QStandardItem(key)
            key_item.setData(value, Qt.UserRole)
            self.appendRow([key_item])

    def remove_item(self, row):
        if 0 <= row < self.rowCount():
            self.removeRow(row)

    def set_selection(self, index):
        self.selection = index
        self.selection_changed.emit(index)
