"""
Module for handling tree representations of data.
"""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QTreeView, QMenu

from pydetecdiv.app import list_tools
from pydetecdiv.app.gui.Trees import TreeDictModel


class ToolboxTreeView(QTreeView):
    """
    A class expanding QTreeView with specific features to view tools and tool categories as a tree.
    """

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
        """
        Launch the currently selected tool 
        """
        selection = self.currentIndex()
        print(selection.internalPointer().item_data)


class ToolboxTreeModel(TreeDictModel):
    """
    A class expanding TreeDictModel with specific features to handle tools and tool categories. This model is populated
    from a dictionary with categories as keys and list of tools as values. The dictionary is return by the list_tools()
    function
    """

    def __init__(self, parent=None):
        super().__init__(list_tools(), ["Tool", "version"], parent=parent)

    def is_category(self, index):
        """
        Check whether the item with this index is a category
        :param index: index of the item to be tested
        :type index: QModelIndex
        :return: True if it is a category, False otherwise
        :rtype: bool
        """
        if index.parent().row() == -1:
            return True
        return False

    def is_tool(self, index):
        """
        Check whether the item with this index is a tool
        :param index: index of the item to be tested
        :type index: QModelIndex
        :return: True if it is a tool, False otherwise
        :rtype: bool
        """
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

        if self.hasChildren(index):
            return Qt.ItemIsEnabled

        return Qt.NoItemFlags
