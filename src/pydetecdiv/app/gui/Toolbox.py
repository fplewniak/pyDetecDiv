"""
Module for handling tree representations of data.
"""
from PySide6.QtCore import Qt, QModelIndex
from PySide6.QtWidgets import QTreeView, QMenu

from pydetecdiv.app.gui.Trees import TreeDictModel


class ToolboxTreeView(QTreeView):
    def __init__(self):
        super().__init__()

    def contextMenuEvent(self, event):
        """
        The context menu for area manipulation
        :param event:
        """
        index = self.currentIndex()
        rect = self.visualRect(index)
        if index and not self.model().is_category(index) and rect.top() <= event.pos().y() <= rect.bottom():
            menu = QMenu()
            launch_tool = menu.addAction("Launch tool")
            launch_tool.triggered.connect(self.launch_tool)
            menu.exec(self.viewport().mapToGlobal(event.pos()))

    def launch_tool(self):
        selection = self.currentIndex()
        print(selection.internalPointer().item_data)


class ToolboxTreeModel(TreeDictModel):
    def __init__(self, data, columns, parent=None):
        super().__init__(data, columns, parent=parent)

    def is_category(self, index):
        if index.parent().row() == -1:
            return True
        return False

    def is_tool(self, index):
        return not self.is_category(index)

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

        if self.is_tool(index):
            return Qt.ItemIsEnabled | Qt.ItemIsSelectable
        elif self.hasChildren(index):
            return Qt.ItemIsEnabled

        return Qt.NoItemFlags
