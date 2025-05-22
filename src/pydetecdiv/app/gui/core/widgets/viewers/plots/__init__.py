import numpy as np
from PySide6.QtCharts import QChartView, QLineSeries, QChart
from PySide6.QtCore import QPointF
from PySide6.QtWidgets import QWidget, QVBoxLayout, QGraphicsWidget
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
import pyqtgraph as pg


class MatplotViewer(QWidget):
    """
    A widget to display matplotlib plots in a tab
    """

    def __init__(self, parent: QWidget = None, rows: int = 1, columns: int = 1, toolbar: bool = True):
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

    def show(self) -> None:
        """
        Show the plot
        """
        self.canvas.draw()


class ChartView(pg.GraphicsLayoutWidget):
    """
    A generic viewer for a Chart using PyQtGraph package extensions to PySide6
    """
    def __init__(self, parent: QWidget = None):
        super().__init__(show=False, parent=parent)
        self.addPlot(row=0, col=0)

    def addPlot(self, row: int = 0, col: int = 0) -> None:
        """
        Adds a plot at the desired position designed by row and column
        :param row: the row index
        :param col: the column index
        """
        self.centralWidget.addPlot(row=row, col=col)

    def chart(self, row: int = 0, col: int = 0) -> pg.PlotItem:
        """
        Return the chart PlotItem at the position designated by row and column
        :param row: the row index
        :param col: the column index
        :return: pg.PlotItem
        """
        return self.centralWidget.getItem(row, col)

    def addLinePlot(self, data: np.ndarray, row: int = 0, col: int = 0, **kwargs) -> None:
        """
        Add a line plot representing data at (row, col) position in the Chart view
        :param data: the data to plot
        :param row: the row index
        :param col: the column index
        :param kwargs: extra kwargs to pass to pg.PlotCurveItem constructor
        """
        self.chart(row, col).addItem(pg.PlotCurveItem(data, **kwargs))

    def addScatterPlot(self, data: np.ndarray, row: int = 0, col: int = 0, **kwargs) -> None:
        """
        Add a scatter plot representing data at (row, col) position in the Chart view
        :param data: the data to plot
        :param row: the row index
        :param col: the column index
        :param kwargs: extra kwargs to pass to pg.PlotCurveItem constructor
        """
        scatter = pg.ScatterPlotItem(**kwargs)
        spots = [(i, c) for i, c in enumerate(data)]
        scatter.addPoints(pos=spots)
        self.chart(row, col).addItem(scatter)
        scatter.sigClicked.connect(self.clicked)

    def addXline(self, x, row=0, col=0, **kwargs) -> None:
        """
        Add a vertical line at position x in a plot designated by row and column indices
        :param x: the x position in the plot
        :param row: the row index
        :param col: the column index
        :param kwargs: extra kwargs to pass to pg.PlotCurveItem constructor
        """
        xline = pg.InfiniteLine(**kwargs)
        xline.setPos([x, 0])
        self.chart(row, col).addItem(xline)

    def clicked(self, plot: pg.ScatterPlotItem, points: list[pg.SpotItem]) -> None:
        """
        Slot that needs to be implemented by subclasses to define behaviour in reaction to mose click
        :param plot: the clicked plot item
        :param points: the clicked point
        """
        pass
