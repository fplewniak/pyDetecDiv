from PySide6.QtCharts import QChartView, QLineSeries, QChart
from PySide6.QtCore import QPointF
from PySide6.QtWidgets import QWidget, QVBoxLayout
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
import pyqtgraph as pg

class MatplotViewer(QWidget):
    """
    A widget to display matplotlib plots in a tab
    """

    def __init__(self, parent=None, rows=1, columns=1, toolbar=True):
        super().__init__(parent)
        self.canvas = FigureCanvas(Figure())
        self.axes = self.canvas.figure.subplots(rows, columns)
        self.canvas.figure.tight_layout()

        vlayout = QVBoxLayout()
        vlayout.addWidget(self.canvas)
        if toolbar:
            self.toolbar = QWidget(self)
            self.matplot_toolbar = NavigationToolbar(self.canvas, self)
            vlayout.addWidget(self.matplot_toolbar)
        self.setLayout(vlayout)

    def show(self):
        self.canvas.draw()


class ChartView(pg.GraphicsLayoutWidget):
    def __init__(self, parent=None):
        super().__init__(show=False, parent=parent)
        self.addPlot()

    def chart(self, row=0, col=0):
        return self.getItem(row, col)

    def addLinePlot(self, data, row=0, col=0, **kwargs):
        self.chart(row, col).addItem(pg.PlotCurveItem(data, **kwargs))

    def addScatterPlot(self, data, row=0, col=0, **kwargs):
        scatter = pg.ScatterPlotItem(**kwargs)
        spots = [(i, c) for i, c in enumerate(data)]
        scatter.addPoints(pos=spots)
        self.chart(row, col).addItem(scatter)
        scatter.sigClicked.connect(self.clicked)

    def clicked(self, plot, points):
        pass
