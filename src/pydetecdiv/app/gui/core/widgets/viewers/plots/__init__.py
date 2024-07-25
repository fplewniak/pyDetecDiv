from PySide6.QtCharts import QChartView, QLineSeries, QChart
from PySide6.QtCore import QPointF
from PySide6.QtWidgets import QWidget, QVBoxLayout
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar

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


class ChartView(QChartView):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setChart(Chart())
        self.chart().legend().hide()


class Chart(QChart):
    def __init__(self):
        super().__init__()

    def plot_line(self, series):
        line_series = QLineSeries()
        for row in series:
            line_series.append(row[0], row[1])
        self.addSeries(line_series)
        self.createDefaultAxes()

