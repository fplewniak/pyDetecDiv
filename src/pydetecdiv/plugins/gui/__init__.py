from PySide6.QtSql import QSqlQueryModel
from PySide6.QtWidgets import QDialog, QFrame, QVBoxLayout, QGroupBox, QFormLayout, QLabel, QDialogButtonBox, \
    QSizePolicy, QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox, QAbstractSpinBox, QTableView, QAbstractItemView

class Dialog(QDialog):
    groupBox_styleSheet = """
                                    QGroupBox {
                                        border: 1px solid lightgray;
                                        border-radius: 3px;
                                        padding-top: 0.5em;
               
                                        padding-bottom: 0.5em;
                                        margin-top: 0.5em;
                                        font-weight: bold;
                                    }
                                    QGroupBox::title {
                                        subcontrol-origin: margin;
                                        subcontrol-position: top center;
                                    }
                                    """

    def __init__(self, plugin):
        super().__init__()
        self.form = QFrame()
        self.form.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)
        self.vert_layout = QVBoxLayout(self.form)
        self.form.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Maximum)
        self.setLayout(self.vert_layout)
        self.plugin = plugin
        self.ok = False

    def addGroupBox(self, title):
        group_box = GroupBox(self.form)
        group_box.setTitle(title)
        group_box.setStyleSheet(self.groupBox_styleSheet)
        return group_box

    def addButtonBox(self, buttons=QDialogButtonBox.Ok|QDialogButtonBox.Close, centered=True):
        button_box = QDialogButtonBox(buttons, self)
        button_box.setCenterButtons(centered)
        return button_box

    def arrangeWidgets(self, widget_list):
        for widget in widget_list:
            self.vert_layout.addWidget(widget)


class GroupBox(QGroupBox):
    def __init__(self, parent):
        super().__init__(parent)
        self.layout = QFormLayout(self)

    def addOption(self, label=None, widget=None, **kwargs):
        option = widget(self, **kwargs)
        if label is None:
            self.layout.addRow(option)
        else:
            self.layout.addRow(QLabel(label), option)
        return option

    def setRowVisible(self, index, on=True):
        self.layout.setRowVisible(index, on)


class ComboBox(QComboBox):
    def __init__(self, parent):
        super().__init__(parent)

    def addItemDict(self, options):
        for label, data in options.items():
            self.addItem(label, userData=data)


class LineEdit(QLineEdit):
    def __init__(self, parent):
        super().__init__(parent)


class SpinBox(QSpinBox):
    def __init__(self, parent, range=(1, 4096), single_step=1, adaptive=False, value=None):
        super().__init__(parent)
        self.setRange(*range[0:2])
        self.setSingleStep(single_step)
        if adaptive:
            self.setStepType(QAbstractSpinBox.AdaptiveDecimalStepType)
        if value is None:
            self.setValue(range[0])
        else:
            self.setValue(value)


class DoubleSpinBox(QDoubleSpinBox):
    def __init__(self, parent, range=(0.1, 1.0), decimals=2, single_step=0.1, adaptive=False, value=0.1, enabled=True):
        super().__init__(parent)
        self.setRange(*range[0:2])
        self.setDecimals(decimals)
        self.setSingleStep(single_step)
        if adaptive:
            self.setStepType(QAbstractSpinBox.AdaptiveDecimalStepType)
        if value is None:
            self.setValue(range[0])
        else:
            self.setValue(value)
        self.setEnabled(enabled)

class TableView(QTableView):
    def __init__(self, parent, multiselection=True, behavior='rows'):
        super().__init__(parent)
        self.model = QSqlQueryModel()
        self.setModel(self.model)
        if multiselection:
            self.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        match behavior:
            case 'rows':
                self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
            case _:
                pass

    def setQuery(self, query):
        self.model.setQuery(query)

class DialogButtonBox(QDialogButtonBox):
    def __init__(self, parent, buttons=QDialogButtonBox.Ok|QDialogButtonBox.Close):
        super().__init__(parent)
        self.addButton(buttons)
