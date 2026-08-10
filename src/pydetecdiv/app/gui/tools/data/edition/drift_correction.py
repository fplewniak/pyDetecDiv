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

        set_connections({self.button_box.accepted       : self.tool.callback,
                         self.button_box.rejected       : self.close,
                         PyDetecDiv.app.project_selected: self.tool.update_fov_list,
                         })

        self.exec()
