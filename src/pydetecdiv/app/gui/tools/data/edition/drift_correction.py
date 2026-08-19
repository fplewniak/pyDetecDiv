"""
Classes for drift correction
"""
from pydetecdiv.app import PyDetecDiv
from pydetecdiv.app.gui.core.widgets import DictListView, set_connections
from pydetecdiv.app.tools import Tool

from pydetecdiv.app.gui.tools import ToolDialog


class ComputeDriftDialog(ToolDialog):
    """
    Dialog window to launch computing of drift
    """

    def __init__(self, tool: Tool):
        super().__init__(tool, title=tool.title)

        self.select_FOV = self.addGroupBox('Select FOVs',
                                           parameters=[self.tool.parameters.FOVs],
                                           widget_args={'FOVs': {'widget': DictListView}}
                                           )

        self.method_box = self.addGroupBox('Method',
                                           parameters=[self.tool.parameters.method],
                                           )

        self.button_box = self.addButtonBox()

        self.arrangeWidgets([
            self.select_FOV,
            self.method_box,
            self.button_box
            ])

        self.run_after_process([self.plot_drift])

        set_connections({self.button_box.accepted: lambda: self.wait_for_command(msg=f'Computing drift, please wait',
                                                                                 cancel_msg='Cancel drift computation please wait'),
                         self.button_box.rejected: self.close,
                         })

        self.exec()

    def plot_drift(self):
        """
        Plots the drift values for the selected FOVs in tabbed windows
        """
        tab = PyDetecDiv.main_window.add_tabbed_window(
                f'{PyDetecDiv.project_name} / Drift correction ({self.tool.parameters.method.value})')
        tab.project_name = PyDetecDiv.project_name
        for fov in self.tool.parameters.FOVs.qmodel.selected_values():
            tab.show_plot(self.tool.drift[fov.name], title=fov.name)
