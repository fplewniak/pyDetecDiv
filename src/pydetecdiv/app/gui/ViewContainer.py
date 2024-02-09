from PySide6.QtWidgets import QMainWindow


class ViewContainer(QMainWindow):
    def __init__(self, **kwargs):
        super().__init__()
        self.project_name = None

    def close_window(self):
        """
        Close the Tabbed viewer containing this Image viewer
        """
        self.parent().parent().window.close()
