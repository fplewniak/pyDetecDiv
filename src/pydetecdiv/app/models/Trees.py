"""
Module for handling tree representations of data.
"""
from typing import Any

from PySide6.QtCore import QAbstractItemModel, Qt, QModelIndex


class TreeItem:
    """
    A tree node
    """

    def __init__(self, data, parent=None):
        self.parent_item = parent
        self.item_data = data
        self.child_items = []

    def append_child(self, item):
        """
        Add a child item to the current node

        :param item: child item
        :type item: TreeItem
        """
        self.child_items.append(item)
        item.parent_item = self

    def child(self, row):
        """
        Return the child at index row

        :param row: the row model index
        :type row: int
        :return: the requested child item
        :rtype: TreeItem
        """
        return self.child_items[row]

    def child_count(self):
        """
        Count the number of children of the node

        :return: the number of node children
        :rtype: int
        """
        return len(self.child_items)

    def column_count(self):
        """
        Count the number of columns in model data

        :return: the number of columns
        :rtype: int
        """
        return len(self.item_data)

    def data(self, column):
        """
        The data for a given column index

        :param column: the column index
        :type column: int
        :return: the node's data in the column index
        :rtype: whatever type is in data
        """
        try:
            return self.item_data[column]
        except IndexError:
            return None

    def set_data(self, column, data):
        self.item_data[column] = data

    def parent(self):
        """
        Return the parent item of the node

        :return: the parent item
        :rtype: TreeItem
        """
        return self.parent_item

    def row(self):
        """
        Return the row index of the node

        :return: the row index
        :rtype: int
        """
        if self.parent_item:
            return self.parent_item.child_items.index(self)

        return 0


class TreeModel(QAbstractItemModel):
    """
    A model to represent a dictionary as a Tree. Dictionary values are lists of leaves or internal nodes.
    Internal nodes are represented as sub-dictionaries and leaves are lists of data items corresponding each to one
    column.
    """

    def __init__(self, columns, data=None, parent=None):
        super().__init__(parent)
        self.root_item = TreeItem(columns)
        self.setup_model_data(data)

    def columnCount(self, parent=QModelIndex()):
        """
        Returns the number of columns for the children of the given parent

        :param parent: the parent index
        :type parent: QModelIndex
        :return: the number of columns
        :rtype: int
        """
        if parent.isValid():
            return parent.internalPointer().column_count()
        return self.root_item.column_count()

    def data(self, index, role=Qt.DisplayRole):
        """
        Returns the data stored under the given role for the item referred to by the index.

        :param index: the node index
        :type index: QModelIndex
        :param role: the role
        :type role: int (Qt.ItemDataRole)
        :return: the data
        :rtype: object
        """
        if not index.isValid():
            return None

        if role != Qt.DisplayRole:
            return None

        item = index.internalPointer()

        return item.data(index.column())

    def flags(self, index):
        """
        Returns the item flags for the given index

        :param index: the index
        :type index: QModelIndex
        :return: the flags
        :rtype: Qt.ItemFlag
        """
        if not index.isValid():
            return Qt.NoItemFlags

        return Qt.ItemIsEnabled | Qt.ItemIsSelectable

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        """
        Returns the data for the given role and section in the header with the specified orientation

        :param section: the section
        :type section: int
        :param orientation: the orientation
        :type orientation: Qt.Orientation
        :param role: the role
        :type role: int (Qt.ItemDataRole)
        :return: data
        :rtype: object
        """
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self.root_item.data(section)

        return None

    def index(self, row, column, parent=QModelIndex()):
        """
        Returns the index of the item in the model specified by the given row, column and parent index.

        :param row: the row
        :type row: int
        :param column: the column
        :type column: int
        :param parent: the parent index
        :type parent: QModelIndex
        :return: the item index
        :rtype: QModelIndex
        """
        if not self.hasIndex(row, column, parent):
            return QModelIndex()

        if not parent.isValid():
            parent_item = self.root_item
        else:
            parent_item = parent.internalPointer()

        child_item = parent_item.child(row)
        if child_item:
            return self.createIndex(row, column, child_item)
        return QModelIndex()

    def parent(self, index=QModelIndex()):
        """
        Returns the parent of the model item with the given index.

        :param index: the index
        :type index: QModelIndex
        :return: parent index
        :rtype: QModelIndex
        """
        if not index.isValid():
            return QModelIndex()

        child_item = index.internalPointer()
        parent_item = child_item.parent()

        if parent_item == self.root_item:
            return QModelIndex()

        return self.createIndex(parent_item.row(), 0, parent_item)

    def rowCount(self, parent=QModelIndex()):
        """
        Returns the number of rows under the given parent. When the parent is valid it means that rowCount is returning
        the number of children of parent.

        :param parent: the parent index
        :type parent: QModelIndex
        :return: the number of rows
        :rtype: int
        """
        if parent.column() > 0:
            return 0

        if not parent.isValid():
            parent_item = self.root_item
        else:
            parent_item = parent.internalPointer()

        return parent_item.child_count()

    def removeRow(self, row, parent=QModelIndex()):
        ...

    def setup_model_data(self, data):
        """
        Set the model up from data

        :param data: the data to load into the model
        :type data: object
        :param parent: the root of the tree
        :type parent: TreeItem
        """


class TreeDictModel(TreeModel):
    """
    A Tree model that can be created from dictionaries
    """

    def __init__(self, columns: list[str], data: dict[str, Any] | None = None, parent=None):
        super().__init__(columns, data=None, parent=parent)
        self.parents = [self.root_item]
        if data is not None:
            self.setup_model_data(data)
            self.data_dict = data
        else:
            self.data_dict = {}

    def setup_model_data(self, data):
        """
        Set the model up from the data stored in a dictionary

        :param data: the dictionary to load into the model
        :type data: dict
        :param parent: the root of the tree
        :type parent: TreeItem
        """
        if data is not None:
            self.append_children(data, self.root_item)

    def append_children(self, data, parent):
        """
        Append children to an arbitrary node represented by a dictionary. This method is called recursively to load the
        successive levels of nodes.

        :param data: the dictionary to load at this node
        :type data: dict
        :param parent: the internal node
        :type parent: TreeItem
        """
        for key, values in data.items():
            self.parents.append(TreeItem([key, ''], parent))
            parent.append_child(self.parents[-1])
            if isinstance(values, dict):
                print(values, ' is dict')
                self.append_children(values, self.parents[-1])
            elif isinstance(values, str):
                print(values, ' is str')
                self.parents[-1].append_child(TreeItem(values, self.parents[-1]))
            else:
                print(values, ' is list')
                for v in values:
                    self.parents[-1].append_child(TreeItem(v, self.parents[-1]))
