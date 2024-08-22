from PySide6.QtCore import QAbstractItemModel, QModelIndex, Qt, QAbstractListModel, QStringListModel, \
    QAbstractTableModel, Signal
from PySide6.QtGui import QStandardItemModel, QStandardItem


class Text(QAbstractItemModel):
    def __init__(self, text: str = '', parent=None):
        super().__init__(parent)
        self._text = text

    @property
    def value(self):
        return self.data(self.index(0, 0))

    def rowCount(self, parent=QModelIndex()):
        return 1  # Une seule ligne

    def columnCount(self, parent=QModelIndex()):
        return 1  # Une seule colonne

    def data(self, index, role=Qt.DisplayRole):
        if role == Qt.DisplayRole or role == Qt.EditRole:
            return self._text
        return None

    def setData(self, index, value, role=Qt.EditRole):
        if role == Qt.EditRole:
            self._text = value
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


class DictList(QStandardItemModel):
    def __init__(self, data_dict=None, parent=None):
        super().__init__(parent)
        self.setColumnCount(2)
        self.setHorizontalHeaderLabels(['label', 'data'])
        if data_dict:
            self.setItemsDict(data_dict)

    def rows(self):
        return {self.item(row, 0).text(): self.item(row, 1).data(Qt.UserRole) for row in range(self.rowCount())}

    def setItemsDict(self, data_dict):
        for key, value in data_dict.items():
            self.addItem(key, value)

    def addItem(self, key, value):
        key_item = QStandardItem(key)
        key_item.setData(value, Qt.UserRole)
        value_item = QStandardItem(str(value))  # Optionnel: convertir l'objet en chaîne pour l'affichage
        # value_item.setData(value, Qt.UserRole)  # Stocker l'objet dans le rôle UserRole
        self.appendRow([key_item, value_item])

    def removeItem(self, row):
        if 0 <= row < self.rowCount():
            self.removeRow(row)

    def clear(self):
        self.clear()
        self.setHorizontalHeaderLabels(['label', 'data'])

    # def getItemObject(self, row):
    #     if 0 <= row < self.rowCount():
    #         return self.item(row, 1).data(Qt.UserRole)
    #     return None
    #
    # def getItemKey(self, row):
    #     if 0 <= row < self.rowCount():
    #         return self.item(row, 0).data(Qt.UserRole)
    #     return None
    #
    # def getData(self, key):
    #     for row in range(self.rowCount()):
    #         if key == self.item(row, 0).text():
    #             return self.item(row, 1).data(Qt.UserRole)
