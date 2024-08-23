from pydetecdiv.app.models import StringList, DictItemModel, ItemModel
from pydetecdiv.plugins import Dialog
from PySide6.QtCore import QModelIndex, QAbstractTableModel, QItemSelectionModel, QItemSelection

from pydetecdiv.plugins.gui import LineEdit, ComboBox


from PySide6.QtCore import QAbstractListModel, Qt

from pydetecdiv.plugins.parameters import Parameter


class TrainingDialog(Dialog):
    def __init__(self, plugin, title=None):
        super().__init__(plugin, title='Training classification model')

        # self.text_model = Parameter(name='text', default='Initial text')
        self.text_model = ItemModel('Initial text')
        # print(self.text_model.value)
        self.line_edit = LineEdit(self, self.text_model)
        # self.mapper = QDataWidgetMapper(self)
        # self.mapper.setModel(self.text_model)
        # self.mapper.addMapping(self.line_edit, 0)
        # self.mapper.toFirst()

        self.items = {
            "Item 1": 'Objet numero 1',
            "Item 2": 'Objet numero 2',
            "Item 3": 'Objet numero 3',
        }
        self.model = DictItemModel(self.items)
        self.combo_box = ComboBox(self, self.model)
        # self.combo_box.setModel(self.model)
        # self.combo_box.setModelColumn(0)

        self.button_box = self.addButtonBox()
        self.button_box.accepted.connect(self.run_training)
        self.button_box.rejected.connect(self.close)

        # self.line_edit.changed.connect(lambda _: print(self.text_model.value))
        # self.combo_box.changed.connect(lambda _: print('changed label', self.model.index(self.combo_box.currentIndex(), 0).data()))
        # self.combo_box.changed.connect(lambda _: print('changed data', self.model.index(self.combo_box.currentIndex(), 1).data(Qt.UserRole)))
        # self.combo_box.changed.connect(lambda _: print('changed value', self.combo_box.value()))
        self.combo_box.changed.connect(lambda _: print('changed value', self.combo_box.value(), self.model.value()))

        self.arrangeWidgets([self.line_edit, self.combo_box, self.button_box])

        self.fit_to_contents()
        self.exec()

    def run_training(self):
        print(self.text_model.value)
        self.model.add_item({"XXX": {'name': 'Object 4', 'value': 500}})
        print(self.combo_box.currentText(), self.combo_box.value())
        self.combo_box.setCurrentText("XXX")
        print(self.combo_box.currentText(), self.combo_box.value())
        v = self.model.value()
        print(v['name'])
        print(self.model.rows())
        # print('label', self.model.index(self.combo_box.currentIndex(), 0).data())
        # print('data', self.model.index(self.combo_box.currentIndex(), 1).data(Qt.UserRole))
        # print(self.combo_box.value())
        # print(self.model.items())

        self.text_model.setData(self.text_model.index(0,0), 'Training model')
        print(self.text_model.value())
        # self.line_edit.setValue('Training model')
        # print(self.text_model.value)


class FineTuningDialog(Dialog):
    def __init__(self, plugin, title=None):
        super().__init__(plugin, title='Fine tuning classification model')



        self.button_box = self.addButtonBox()
        self.button_box.accepted.connect(self.run_fine_tuning)
        self.button_box.rejected.connect(self.close)

        self.arrangeWidgets([self.button_box])

        self.fit_to_contents()
        self.exec()

    def run_fine_tuning(self):
        print('Fine tuning model')
