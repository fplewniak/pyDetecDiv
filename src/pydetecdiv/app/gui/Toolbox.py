"""
Module for handling tree representations of data.
"""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QTreeView, QMenu, QDialogButtonBox, QDialog

from pydetecdiv.app import list_tools
from pydetecdiv.app.gui.Trees import TreeDictModel, TreeItem


class ToolItem(TreeItem):
    def __init__(self, data, parent=None):
        super().__init__(data, parent=parent)
        self.tool = data
        self.item_data = [data.name, data.version]

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
        tool_form = ToolForm(selection.internalPointer().tool)
        tool_form.exec()
        # print(selection.internalPointer().item_data)
        # print(selection.internalPointer().tool.categories)
        # print(selection.internalPointer().tool.attributes)
        # print(selection.internalPointer().tool.command)
        # selection.internalPointer().tool.requirements.install()


class ToolForm(QDialog):
    def __init__(self, tool, parent=None):
        super().__init__(parent)
        self.tool = tool
        self.OK_button = QDialogButtonBox(self)
        self.OK_button.setObjectName("OK_button")
        self.OK_button.setStandardButtons(QDialogButtonBox.Ok)
        self.OK_button.accepted.connect(self.accept)
        # self.addWidget(self.OK_button)

    def accept(self):
        print('running job')
        # job = Run(...)
        # self.tool.requirements.install()
        # job.run()


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
                self.append_children(values, self.parents[-1])
            else:
                for v in values:
                    self.parents[-1].append_child(ToolItem(v, self.parents[-1]))
