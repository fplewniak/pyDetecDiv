"""
Handling actions pertaining to deep learning
"""
from typing import Any

from PySide6.QtGui import QAction
from PySide6.QtWidgets import QWidget
from torchinfo import summary

from pydetecdiv.app.gui.core.widgets import Dialog, set_connections
from pydetecdiv.app.parameters import ChoiceParameter, IntParameter, Parameters
from pydetecdiv.domain.tools.video_classifier.models.MViT import MViT_v2_s, MViT_v1_b


class CreateClassificationSchemeAction(QAction):
    """
    Action to open a shared data source configuration window
    """

    def __init__(self, parent: QWidget):
        super().__init__("Create classification scheme", parent)
        self.triggered.connect(lambda _: print('Create classification scheme'))
        parent.addAction(self)


class ShowModelInfoDialog(Dialog):
    def __init__(self, title: str = None, **kwargs: dict[str, Any]) -> None:
        super().__init__(title, **kwargs)
        self.parameters = Parameters(
                [
                    ChoiceParameter('model', items={'MViT_v2_small': MViT_v2_s,'MViT_v1_b'    : MViT_v1_b,}, label='Model'),
                    IntParameter('num_classes', label='Number of classes', default=6),
                    IntParameter('batch_size', label='Batch size', default=8)
                    ]
                )

        model_choice = self.addGroupBox(
                parameters=[
                    self.parameters.model,
                    self.parameters.num_classes,
                    self.parameters.batch_size
                    ],
                )
        button_box = self.addButtonBox()

        self.arrangeWidgets([
            model_choice,
            button_box
            ])

        set_connections({
            button_box.accepted: self.show_model_information,
            })

        self.fit_to_contents()
        self.exec()

    def show_model_information(self):
        model = self.parameters.model.value(n_classes=self.parameters.num_classes.value)
        summary(model, (self.parameters.batch_size.value, 15, 3, 224, 224), device='cpu')


class ShowModelInformationAction(QAction):
    """
    Action to open a shared data source configuration window
    """

    def __init__(self, parent: QWidget):
        super().__init__("Show model information", parent)
        self.triggered.connect(ShowModelInfoDialog)
        parent.addAction(self)
