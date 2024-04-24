from PySide6.QtGui import QIcon
from PySide6.QtSql import QSqlQueryModel
from PySide6.QtWidgets import QDialog, QFrame, QVBoxLayout, QGroupBox, QFormLayout, QLabel, QDialogButtonBox, \
    QSizePolicy, QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox, QAbstractSpinBox, QTableView, QAbstractItemView, \
    QPushButton, QApplication, QRadioButton


class StyleSheets:
    groupBox = """
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


class GroupBox(QGroupBox):
    def __init__(self, parent, title=None):
        super().__init__(parent)
        if title is not None:
            self.setTitle(title)
        self.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Maximum)


class FormGroupBox(GroupBox):
    def __init__(self, parent, title=None, show=True):
        super().__init__(parent, title)
        self.layout = QFormLayout(self)
        self.setVisible(show)

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
    def __init__(self, parent, items=None, selected=None):
        super().__init__(parent)
        if items is not None:
            self.addItemDict(items)
        if selected is not None:
            self.setCurrentText(selected)

    def addItemDict(self, options):
        for label, data in options.items():
            self.addItem(label, userData=data)

    @property
    def selected(self):
        return self.currentIndexChanged

    @property
    def changed(self):
        return self.currentTextChanged


class LineEdit(QLineEdit):
    def __init__(self, parent):
        super().__init__(parent)


class PushButton(QPushButton):
    def __init__(self, parent, text, icon=None, flat=False):
        if icon is None:
            super().__init__(text, parent)
        else:
            super().__init__(icon, text, parent)
        self.setFlat(flat)


class AdvancedButton(PushButton):
    def __init__(self, parent):
        super().__init__(parent, text='Advanced options', icon=QIcon(':icons/show'), flat=True)
        self.group_box = None
        self.clicked.connect(self.toggle)

    def hide(self):
        super().hide()
        self.setIcon(QIcon(':icons/show'))
        self.group_box.setVisible(False)

    def linkGroupBox(self, group_box):
        self.group_box = group_box

    def toggle(self):
        if self.group_box.isVisible():
            self.setIcon(QIcon(':icons/show'))
            self.group_box.setVisible(False)
        else:
            self.setIcon(QIcon(':icons/hide'))
            self.group_box.setVisible(True)
        self.parent().parent().fit_to_contents()

class RadioButton(QRadioButton):
    def __init__(self, parent, exclusive=True):
        super().__init__(None, parent)
        self.setAutoExclusive(exclusive)

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

    @property
    def changed(self):
        return self.valueChanged


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

    @property
    def changed(self):
        return self.valueChanged


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
    def __init__(self, parent, buttons=(QDialogButtonBox.Ok, QDialogButtonBox.Close)):
        super().__init__(parent)
        for button in buttons:
            self.addButton(button)

    def connect_to(self, connections=None):
        if connections is not None:
            for signal, slot in connections.items():
                match signal:
                    case 'accept':
                        self.accepted.connect(slot)
                    case 'reject':
                        self.rejected.connect(slot)
                    case 'click':
                        self.clicked.connect(slot)
                    case 'help':
                        self.helpRequested.connect(slot)


class Dialog(QDialog):
    def __init__(self, plugin, title=None):
        super().__init__()
        self.vert_layout = QVBoxLayout(self)
        self.setLayout(self.vert_layout)
        self.plugin = plugin
        if title is not None:
            self.setWindowTitle(title)
        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Maximum)

    def fit_to_contents(self):
        QApplication.processEvents()
        self.adjustSize()

    def addGroupBox(self, title, widget=FormGroupBox):
        group_box = widget(self)
        group_box.setTitle(title)
        group_box.setStyleSheet(StyleSheets.groupBox)
        return group_box

    def addButtonBox(self, buttons=QDialogButtonBox.Ok | QDialogButtonBox.Close, centered=True):
        button_box = DialogButtonBox(self, buttons=buttons)
        button_box.setCenterButtons(centered)
        return button_box

    def addButton(self, widget, text=None, icon=None, flat=False):
        button = widget(text, icon)
        if flat:
            button.setFlat(True)
        return button

    def arrangeWidgets(self, widget_list):
        for widget in widget_list:
            self.vert_layout.addWidget(widget)


def set_connections(connections):
    for signal, slot in connections.items():
        if isinstance(slot, list):
            for s in slot:
                signal.connect(s)
        else:
            signal.connect(slot)
