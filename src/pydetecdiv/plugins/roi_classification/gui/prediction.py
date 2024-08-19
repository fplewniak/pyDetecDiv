from pydetecdiv.plugins import Dialog


class PredictionDialog(Dialog):
    def __init__(self, plugin, title=None):
        super().__init__(plugin, title='Predict classes')

        self.button_box = self.addButtonBox()
        self.button_box.accepted.connect(self.run_prediction)
        self.button_box.rejected.connect(self.close)

        self.arrangeWidgets([self.button_box])

        self.fit_to_contents()
        self.exec()

    def run_prediction(self):
        print('running prediction')
