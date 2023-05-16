"""
Module for handling tree representations of data.
"""
from PySide6.QtCore import QAbstractItemModel, Qt, QModelIndex
from PySide6.QtWidgets import QTreeView, QMenu

from pydetecdiv.app.gui.Trees import TreeDictModel


class ToolboxTreeView(QTreeView):
    def __init__(self):
        super().__init__()

    def keyPressEvent(self, event):
        print(self.selectedIndexes()[0].data(), self.selectedIndexes()[1].data())

    def contextMenuEvent(self, event):
        """
        The context menu for area manipulation
        :param event:
        """
        if self.selectedIndexes():
            menu = QMenu()
            launch_tool = menu.addAction("Launch tool")
            launch_tool.triggered.connect(self.show_selection)
            menu.exec(self.viewport().mapToGlobal(event.pos()))

    def show_selection(self):
        selection = self.selectedIndexes()
        print([s.data() for s in selection])
        print([s.sibling(s.row(), c).data() for c, s in enumerate(selection)])


class ToolboxTreeModel(TreeDictModel):
    def __init__(self, data, columns, parent=None):
        super().__init__(data, columns, parent=parent)

    def is_category(self, index):
        if index.parent().row() == -1:
            return True
        return False

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

        if self.is_category(index) and self.hasChildren(index):
            return Qt.ItemIsEnabled

        if not self.is_category(index):
            return Qt.ItemIsEnabled | Qt.ItemIsSelectable

        return Qt.NoItemFlags
