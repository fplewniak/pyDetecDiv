from typing import Any

import polars

from pydetecdiv.app import pydetecdiv_project, PyDetecDiv
from pydetecdiv.app.gui.core.widgets import set_connections, TableView
from pydetecdiv.app.models import TableModel
from pydetecdiv.app.tools import Tool
from pydetecdiv.app.gui.tools import ToolDialog


class ManageClassificationSchemeDialog(ToolDialog):
    def __init__(self, tool: Tool, title: str = None, **kwargs: dict[str, Any]) -> None:
        super().__init__(tool, title, **kwargs)

        with pydetecdiv_project(PyDetecDiv.project_name) as project:
            data = project.get_polars('Classification')
            # data = polars.DataFrame(classification_schemes[0].record())
            # print(classification_schemes[0].record())
            # for classification_scheme in classification_schemes[1:]:
            #     data.extend(polars.DataFrame(classification_scheme.record()))
        self.data_view = TableView(self, TableModel(data))
        self.data_view.verticalHeader().setVisible(False)

        classification_management = self.addGroupBox(
                parameters=[
                    self.tool.parameters.name,
                    ],
                )

        button_box = self.addButtonBox()

        self.arrangeWidgets([
            self.data_view,
            classification_management,
            button_box
            ])

        set_connections({
            button_box.accepted: lambda: print(self.tool.parameters),
            })

        self.fit_to_contents()
        self.exec()
