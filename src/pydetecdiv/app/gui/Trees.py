from PySide6.QtCore import QAbstractItemModel, Qt, QModelIndex


class TreeItem(object):
    """
    A tree node
    """
    def __init__(self, data, parent=None):
        self.parentItem = parent
        self.itemData = data
        self.childItems = []

    def appendChild(self, item):
        """
        Add a child item to the current node
        :param item: child item
        :type item: TreeItem
        """
        self.childItems.append(item)

    def child(self, row):
        """
        Return the child at index row
        :param row: the row model index
        :type row: int
        :return: the requested child item
        :rtype: TreeItem
        """
        return self.childItems[row]

    def childCount(self):
        """
        Count the number of children of the node
        :return: the number of node children
        :rtype: int
        """
        return len(self.childItems)

    def columnCount(self):
        """
        Count the number of columns in model data
        :return: the number of columns
        :rtype: int
        """
        return len(self.itemData)

    def data(self, column):
        """
        The data for a given column index
        :param column: the column index
        :type column: int
        :return: the node's data in the column index
        :rtype: whatever type is in data
        """
        try:
            return self.itemData[column]
        except IndexError:
            return None

    def parent(self):
        """
        Return the parent item of the node
        :return: the parent item
        :rtype: TreeItem
        """
        return self.parentItem

    def row(self):
        """
        Return the row index of the node
        :return: the row index
        :rtype: int
        """
        if self.parentItem:
            return self.parentItem.childItems.index(self)

        return 0


class TreeModel(QAbstractItemModel):
    """
    A model to represent a dictionary as a Tree. Dictionary values are lists of leaves or internal nodes.
    Internal nodes are represented as sub-dictionaries and leaves are lists of data items corresponding each to one
    column.
    """
    def __init__(self, data, columns, parent=None):
        super(TreeModel, self).__init__(parent)

        self.rootItem = TreeItem(columns)
        self.setupModelData(data, self.rootItem)

    def columnCount(self, parent):
        """
        Count the number of columns
        :param parent:
        :return:
        """
        if parent.isValid():
            return parent.internalPointer().columnCount()
        else:
            return self.rootItem.columnCount()

    def data(self, index, role):
        """

        :param index:
        :param role:
        :return:
        """
        if not index.isValid():
            return None

        if role != Qt.DisplayRole:
            return None

        item = index.internalPointer()

        return item.data(index.column())

    def flags(self, index):
        """

        :param index:
        :return:
        """
        if not index.isValid():
            return Qt.NoItemFlags

        return Qt.ItemIsEnabled | Qt.ItemIsSelectable

    def headerData(self, section, orientation, role):
        """

        :param section:
        :param orientation:
        :param role:
        :return:
        """
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self.rootItem.data(section)

        return None

    def index(self, row, column, parent):
        """

        :param row:
        :param column:
        :param parent:
        :return:
        """
        if not self.hasIndex(row, column, parent):
            return QModelIndex()

        if not parent.isValid():
            parentItem = self.rootItem
        else:
            parentItem = parent.internalPointer()

        childItem = parentItem.child(row)
        if childItem:
            return self.createIndex(row, column, childItem)
        else:
            return QModelIndex()

    def parent(self, index):
        """

        :param index:
        :return:
        """
        if not index.isValid():
            return QModelIndex()

        childItem = index.internalPointer()
        parentItem = childItem.parent()

        if parentItem == self.rootItem:
            return QModelIndex()

        return self.createIndex(parentItem.row(), 0, parentItem)

    def rowCount(self, parent):
        """
        Count the number of rows in the model
        :param parent:
        :type parent: QModelIndex
        :return: int
        """
        if parent.column() > 0:
            return 0

        if not parent.isValid():
            parentItem = self.rootItem
        else:
            parentItem = parent.internalPointer()

        return parentItem.childCount()

    def setupModelData(self, branches, parent):
        """
        Setup the model from the data stored in a dictionary
        :param branches: the dictionary to load into the model
        :type branches: dict
        :param parent: the root of the tree
        :type parent: TreeItem
        """
        self.parents = [parent]
        self.appendChildren(branches, parent)

    def appendChildren(self, branches, parent):
        """
        Append children to an arbitrary node represented by a dictionary. This method is called recursively to load the
        successive levels of nodes.
        :param branches: the dictionary to load at this node
        :type branches: dict
        :param parent: the internal node
        :type parent: TreeItem
        """
        for key, values in branches.items():
            print(key)
            self.parents.append(TreeItem([key, ''], parent))
            parent.appendChild(self.parents[-1])
            if isinstance(values, dict):
                self.appendChildren(values, self.parents[-1])
            else:
                for v in values:
                    self.parents[-1].appendChild(TreeItem(v, self.parents[-1]))
