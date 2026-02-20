"""
The QActions for the VideoClassifier tool
"""
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QWidget


class TrainModelAction(QAction):
    """
    Action to import raw data images into a project
    """

    def __init__(self, parent: QWidget):
        super().__init__("Train model", parent)
        self.triggered.connect(self.run_training)
        self.setEnabled(False)
        parent.addAction(self)

    def run_training(self):
        """
        Run training procedure
        """
