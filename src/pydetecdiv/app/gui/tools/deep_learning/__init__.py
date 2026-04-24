"""
GUI functions and classes specific to Deep Learning tool GUI
"""
import matplotlib
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QGridLayout
import pyqtgraph as pg

from pydetecdiv.app import PyDetecDiv
from pydetecdiv.app.gui.core import Colours
from pydetecdiv.app.gui.core.widgets.viewers.plots import MatplotViewer, ChartView
from pydetecdiv.torch import ClassifierTrainingStats, TrainingHistory
from pydetecdiv.torch.metrics import is_single_value_metric


def plot_training_results(train_stats: ClassifierTrainingStats) -> None:
    """
    Plots training results (history, confusion matrix, ...)

    :param train_stats: the statistics from training process
    """
    module_name, history = train_stats.model_name, train_stats.history
    tab = PyDetecDiv.main_window.add_tabbed_window(f'{PyDetecDiv.project_name} / {module_name}')
    tab.project_name = PyDetecDiv.project_name
    history_plot = plot_history(history)
    tab.addTab(history_plot, 'Training history')
    tab.setCurrentWidget(history_plot)

    interactive = plot_interactive_history(train_stats)
    tab.addTab(interactive, 'Interactive history')

def plot_interactive_history(train_stats: ClassifierTrainingStats) -> QFrame:
    """
    Create a QFrame to display the interactive history in: the left panel shows the training and validation confusion matrices at
    a given epoch, and the right panel shows the metrics history with a movable vertical line used to select the epoch to display
    the confusion matrices for.

    :param train_stats: the training statistics
    :return: the QFrame
    """
    history = train_stats.history

    frame = QFrame()
    layout = QGridLayout()
    frame.setLayout(layout)

    chart_view, epoch_line = plot_metrics_history(train_stats)

    confusion_matrices_view = MatplotViewer(columns=2, rows=2, toolbar=False)
    update_heat_maps(confusion_matrices_view, train_stats, epoch=history.best_epoch)

    layout.addWidget(confusion_matrices_view, 0, 0)
    layout.addWidget(chart_view, 0, 1)

    epoch_line.sigPositionChangeFinished.connect(
        lambda x: update_heat_maps(confusion_matrices_view, train_stats, epoch=int(x.getXPos() + 0.5)))
    epoch_line.sigPositionChangeFinished.connect(lambda x: x.setPos(float(int(x.getXPos() + 0.5))))

    return frame


def update_heat_maps(matplot_view, train_stats, epoch=None):
    """
    Plots the heatmaps in matplot_view for the requested epoch, based on history in train_stats

    :param matplot_view: the MatplotViewer
    :param train_stats: the training statistics
    :param epoch: the requested epoch
    """
    if epoch is None:
        epoch = train_stats.history.best_epoch
    history = train_stats.history
    class_names = train_stats.class_names

    ax = matplot_view.axes

    plot_heatmap(ax[0][0], history.metric_history('ConfusionMatrix_precision')[epoch].numpy(),
                 class_names=class_names, title='precision (train)')
    plot_heatmap(ax[0][1], history.metric_history('ConfusionMatrix_recall')[epoch].numpy(),
                 class_names=class_names, title='recall (train)')
    plot_heatmap(ax[1][0], history.val_metric_history('ConfusionMatrix_precision')[epoch].numpy(),
                 class_names=class_names, title='precision (val)')
    plot_heatmap(ax[1][1], history.val_metric_history('ConfusionMatrix_recall')[epoch].numpy(),
                 class_names=class_names, title='recall (val)')

    matplot_view.show()

def plot_heatmap(axis: matplotlib.axes.Axes, data: np.ndarray, class_names: list[str] = None, title: str | None = None):
    """
    Plot a heatmap in matplotlib axis for classes named in class_names and defined by data

    :param axis: the axis to plot the heatmap in
    :param data: the data defining the heatmap
    :param class_names: the class names
    :param title: the title for the plot
    """
    axis.clear()
    axis.set_title(title, size=10)
    axis.set_xticks(range(len(class_names)), labels=class_names, rotation=45, ha="right", rotation_mode="anchor", size=8)
    axis.set_yticks(range(len(class_names)), labels=class_names, size=8)
    axis.set_xlabel("Predicted classes", size=8)
    axis.set_ylabel("True classes", size=8)

    axis.imshow(data)
    for i, row in enumerate(data):
        for j, _ in enumerate(row):
            _ = axis.text(j, i, f'{data[i, j].item():0.2f}', ha="center", va="center", color="w", size=8)

def plot_history(history: TrainingHistory) -> MatplotViewer:
    """
    Plots metrics history.

    :param history: metrics history to plot
    :param evaluation: metrics from model evaluation on test dataset, shown as horizontal dashed lines on the plots
    """
    plot_viewer = MatplotViewer(PyDetecDiv.main_window.active_subwindow, columns=2, rows=1)
    axs = plot_viewer.axes
    history.plot_metric(axs[0], history.main_metric)
    axs[0].axhline(history.val_metric_history(history.main_metric)[history.best_epoch], color='red', linestyle='--')
    axs[0].axvline(history.best_epoch, color='red', linestyle='--')
    history.plot(axs[1])
    axs[1].axhline(min(history.val_loss), color='red', linestyle='--')
    axs[1].axvline(history.best_epoch, color='red', linestyle='--')

    plot_viewer.show()
    return plot_viewer

def plot_metrics_history(train_stats: ClassifierTrainingStats, epoch: int | None = None) -> tuple[ChartView, pg.InfiniteLine]:
    """
    Creates a ChartView to display the interactive metrics history

    :param train_stats: the training statistics
    :param epoch: the epoch number to initially place the line marking the current epoch
    :return: the ChartView and the epoch_line objects
    """
    history = train_stats.history
    epoch = train_stats.history.best_epoch if epoch is None else epoch

    chart_view = ChartView()
    chart_view.ci.layout.setSpacing(0)
    chart_view.ci.setContentsMargins(0, 0, 0, 0)

    chart_view.addPlot(0, 0, title='Metrics history')
    chart_view.chart(0, 0).addLegend(offset=(-1, -1), anchor=(0, 0), pen=pg.mkPen('k', width=1),
                                     brush=pg.mkBrush('w'))

    i = 0
    for metric_name in train_stats.metrics.keys():
        # if len(history.metric_history(metric_name)[0].shape) <= 1:
        if is_single_value_metric(metric_name):
            chart_view.addLinePlot(history.metric_history(metric_name), row=0, col=0,
                                   pen=pg.mkPen(Colours.palette[i], width=2), name=metric_name)
            chart_view.addLinePlot(history.val_metric_history(metric_name), row=0, col=0,
                                   pen=pg.mkPen(Colours.palette[i], width=2, style=Qt.DashLine))
            scatter = pg.ScatterPlotItem(size=10, brush=pg.mkBrush(Colours.palette[i]))
            best_epoch, best_val = train_stats.best_value(metric_name)
            scatter.addPoints([best_epoch], [best_val])
            chart_view.chart(0, 0).addItem(scatter)
            i += 1
    ticks = [(float(idx), str(idx + 1)) for idx in range(history.num_epochs)]
    chart_view.chart(0, 0).getAxis('bottom').setTicks([ticks, []])

    epoch_line = chart_view.chart(0, 0).addLine(x=epoch, movable=True, pen=pg.mkPen('g', width=3))
    epoch_line.setBounds((0, history.num_epochs - 1))
    return chart_view, epoch_line
