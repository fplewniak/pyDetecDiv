from pydetecdiv.plugins import Dialog


class TrainingDialog(Dialog):
    def __init__(self, plugin, title=None):
        super().__init__(plugin, title='Training classification model')

        self.button_box = self.addButtonBox()
        self.button_box.accepted.connect(self.run_training)
        self.button_box.rejected.connect(self.close)

        self.arrangeWidgets([self.button_box])

        self.fit_to_contents()
        self.exec()

    def run_training(self):
        print('Training model')


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
