from PySide6.QtWidgets import QGraphicsView, QGraphicsScene


class GraphicsView(QGraphicsView):
    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.setScene(Scene())
        # sizePolicy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        # sizePolicy.setHorizontalStretch(0)
        # sizePolicy.setVerticalStretch(0)
        # sizePolicy.setHeightForWidth(self.sizePolicy().hasHeightForWidth())
        # self.setSizePolicy(sizePolicy)


class Scene(QGraphicsScene):
    def __init__(self, **kwargs):
        super().__init__()
