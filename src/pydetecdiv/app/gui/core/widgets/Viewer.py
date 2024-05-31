from PySide6.QtGui import QColor
from PySide6.QtWidgets import QGraphicsView, QGraphicsScene, QSizePolicy, QWidget, QVBoxLayout
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar


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
        rect0 = self.addRect(0, 0, 20, 20, QColor(0, 125, 255))
        rect1 = self.addRect(-10, -10, 20, 20, QColor(180, 180, 0))
        rect2 = self.addRect(0, 0, 100, 100, QColor(255, 125, 0))
        text = self.addText('This is text')
        text.setParentItem(rect2)
        text.setPos(50, 50)


class MatplotViewer(QWidget):
    """
    A widget to display matplotlib plots in a tab
    """

    def __init__(self, parent=None, rows=1, columns=1):
        super().__init__(parent)
        # self.dismiss_button = QPushButton('Dismiss')
        # self.dismiss_button.clicked.connect(lambda: self.parent().removeWidget(self))
        self.canvas = FigureCanvas(Figure())
        self.axes = self.canvas.figure.subplots(rows, columns)
        self.canvas.figure.tight_layout()
        self.toolbar = QWidget(self)
        self.matplot_toolbar = NavigationToolbar(self.canvas, self)

        # hlayout = QHBoxLayout()
        # hlayout.addWidget(self.matplot_toolbar)
        # hlayout.addWidget(self.dismiss_button)
        # self.toolbar.setLayout(hlayout)

        vlayout = QVBoxLayout()
        vlayout.addWidget(self.canvas)
        vlayout.addWidget(self.matplot_toolbar)
        # vlayout.addWidget(self.toolbar)
        self.setLayout(vlayout)
