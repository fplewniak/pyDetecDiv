"""
Module for handling tree representations of data.
"""
from subprocess import CalledProcessError

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QTreeView, QMenu, QDialogButtonBox, QDialog, QPushButton, QVBoxLayout, QFormLayout

from pydetecdiv.app.gui.parameters import ParameterWidgetFactory

from pydetecdiv.app import PyDetecDiv, pydetecdiv_project, WaitDialog
from pydetecdiv.app.gui.Trees import TreeDictModel, TreeItem
from pydetecdiv.domain.Run import Run
from pydetecdiv.domain.Tool import list_tools


class ToolItem(TreeItem):
    """
    A tool-specific tree item
    """

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
    """
    A form to define input and parameters for running a tool job
    """
    finished = Signal(bool)

    def __init__(self, tool, parent=None):
        super().__init__(parent)
        self.tool = tool
        self.layout = QFormLayout(self)
        self.param_widgets = {name: ParameterWidgetFactory().create(p, parent=self, layout=self.layout) for name, p in
                              self.tool.parameters.items()}

        self.button_box = QDialogButtonBox(self)
        self.run_button = QPushButton(self.button_box)
        self.run_button.setObjectName("run_button")
        self.run_button.setText('Run')
        self.run_button.clicked.connect(self.run)
        self.test_button = QPushButton(self.button_box)
        self.test_button.setObjectName("test_button")
        self.test_button.setText('Test')
        self.test_button.clicked.connect(lambda _: self.run(testing=True))
        self.button_box.addButton(self.test_button, QDialogButtonBox.AcceptRole)
        self.button_box.addButton(self.run_button, QDialogButtonBox.AcceptRole)

        self.layout.addWidget(self.button_box)


    def run(self, testing=False):
        """
        Accept the form, run the job and open a dialog waiting for the job to finish
        """
        for w in self.param_widgets.values():
            w.set_value()
        wait_dialog = WaitDialog(f'Running {self.tool.name}. Please wait, this may take a long time.', self, )
        self.finished.connect(wait_dialog.close_window)
        wait_dialog.wait_for(self.run_job, testing=testing)
        self.close()

    def run_job(self, testing=False):
        """
        Run the job within the context of the currently open project
        """
        with pydetecdiv_project(PyDetecDiv().project_name) as project:
            job = Run(self.tool, project=project)
            if testing:
                job.test()
                project.cancel()
            else:
                try:
                    self.tool.init_dso_inputs(project=project)
                    job.execute()
                except CalledProcessError:
                    project.cancel()
                else:
                    project.save(job)
            self.finished.emit(True)


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
