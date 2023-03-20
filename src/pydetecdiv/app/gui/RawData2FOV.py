import os
import random
import re

import pandas
from PySide6.QtCore import Signal, QThread
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QDialog, QColorDialog, QDialogButtonBox

from pydetecdiv.app.gui.ui.RawData2FOV import Ui_RawData2FOV
from pydetecdiv.app import PyDetecDiv, pydetecdiv_project, WaitDialog


class RawData2FOV(QDialog, Ui_RawData2FOV):

    finished = Signal(bool)
    progress = Signal(int)

    def __init__(self):
        """ Initialization
        Parameters
        ----------
        """
        # Base class
        QDialog.__init__(self, PyDetecDiv().main_window)

        # Initialize the UI widgets
        self.ui = Ui_RawData2FOV()
        self.colours = {
            'FOV': QColor.fromRgb(255, 125, 0, 255),
            'C': QColor.fromRgb(0, 255, 0, 255),
            'T': QColor.fromRgb(0, 255, 255, 255),
            'Z': QColor.fromRgb(255, 255, 0, 255),
        }
        with pydetecdiv_project(PyDetecDiv().project_name) as project:
            raw_data_urls = [d.url for d in project.get_objects('Data')]
            self.samples_text = random.sample(raw_data_urls, min([len(raw_data_urls), 5]))
        self.samples = []
        self.ui.setupUi(self)
        self.samples = [self.ui.sample1, self.ui.sample2, self.ui.sample3, self.ui.sample4, self.ui.sample5][
                       :min([len(raw_data_urls), 5])]
        self.controls = {'FOV': self.ui.pos_check,
                         'C': self.ui.c_check,
                         'T': self.ui.t_check,
                         'Z': self.ui.z_check
                         }
        for i, label_text in enumerate(self.samples_text):
            self.samples[i].setText(label_text)
        self.setWindowTitle('Create FOVs from raw data files')
        self.reset()
        with pydetecdiv_project(PyDetecDiv().project_name) as project:
            annotation_pattern = project.raw_dataset.pattern
        if annotation_pattern:
            wait_dialog = WaitDialog('Creating Fields of view', self, cancel_msg='Cancel FOV creation: please wait',
                                         progress_bar=True, )
            self.finished.connect(wait_dialog.close_window)
            self.progress.connect(wait_dialog.show_progress)
            wait_dialog.wait_for(self.create_fov_annotate, annotation_pattern)
            # self.create_fov_annotate(annotation_pattern)
        else:
            self.exec()
        for child in self.children():
            child.deleteLater()
        self.destroy(True)

    def reset(self):
        self.ui.pos_check.setChecked(True)
        self.ui.c_check.setChecked(False)
        self.ui.t_check.setChecked(False)
        self.ui.z_check.setChecked(False)
        self.colours = {
            'FOV': QColor.fromRgb(255, 125, 0, 255),
            'C': QColor.fromRgb(0, 255, 0, 255),
            'T': QColor.fromRgb(0, 255, 255, 255),
            'Z': QColor.fromRgb(255, 255, 0, 255),
        }
        self.ui.pos_left.setCurrentText(u"position")
        self.ui.pos_pattern.setCurrentText(u"\\d+")
        self.ui.pos_right.setCurrentText('')

        self.ui.c_left.setCurrentText(u"channel")
        self.ui.c_pattern.setCurrentText(u"\\d+")
        self.ui.c_right.setCurrentText('')

        self.ui.t_left.setCurrentText(u"time")
        self.ui.t_pattern.setCurrentText(u"\\d+")
        self.ui.t_right.setCurrentText('')

        self.ui.z_left.setCurrentText(u"_z")
        self.ui.z_pattern.setCurrentText(u"\\d+")
        self.ui.z_right.setCurrentText('')

        self.show_chosen_colours()
        self.change_sample_style()

    def get_regex(self):
        regex = {}
        if self.ui.pos_check.isChecked():
            patterns = [self.ui.pos_left.currentText(),
                        self.ui.pos_pattern.currentText(),
                        self.ui.pos_right.currentText()]
            for i, _ in enumerate(patterns):
                while patterns[i].endswith('\\'):
                    patterns[i] = patterns[i][:-1]
            regex['FOV'] = f'({patterns[0]})(?P<FOV>{patterns[1]})({patterns[2]})'

        if self.ui.c_check.isChecked():
            patterns = [self.ui.c_left.currentText(),
                        self.ui.c_pattern.currentText(),
                        self.ui.c_right.currentText()]
            for i, _ in enumerate(patterns):
                while patterns[i].endswith('\\'):
                    patterns[i] = patterns[i][:-1]
            regex['C'] = f'({patterns[0]})(?P<C>{patterns[1]})({patterns[2]})'

        if self.ui.t_check.isChecked():
            patterns = [self.ui.t_left.currentText(),
                        self.ui.t_pattern.currentText(),
                        self.ui.t_right.currentText()]
            for i, _ in enumerate(patterns):
                while patterns[i].endswith('\\'):
                    patterns[i] = patterns[i][:-1]
            regex['T'] = f'({patterns[0]})(?P<T>{patterns[1]})({patterns[2]})'

        if self.ui.z_check.isChecked():
            patterns = [self.ui.z_left.currentText(),
                        self.ui.z_pattern.currentText(),
                        self.ui.z_right.currentText()]
            for i, _ in enumerate(patterns):
                while patterns[i].endswith('\\'):
                    patterns[i] = patterns[i][:-1]
            regex['Z'] = f'({patterns[0]})(?P<Z>{patterns[1]})({patterns[2]})'
        return regex

    def change_sample_style(self):
        self.clear_colours()
        regex = self.get_regex()
        if regex:
            self.colourize_matches(self.find_matches(regex))

    def find_matches(self, regexes):
        matches = {}
        for what in regexes:
            pattern = re.compile(regexes[what])
            matches[what] = [re.search(pattern, label_text) for label_text in self.samples_text]
        return matches

    @staticmethod
    def get_match_spans(matches, group):
        return {what: [RawData2FOV.get_match_span(match, group) for match in matches[what]] for what in matches}

    @staticmethod
    def get_match_span(match, group=2):
        if match:
            return match.span(group)
        return None

    def colourize_matches(self, matches):
        df = pandas.DataFrame.from_dict(self.get_match_spans(matches, 0))
        columns = set(df.columns)
        conflicting_columns = set()
        for col in columns:
            self.controls[col].setStyleSheet("")
        for i, file_name in enumerate(self.samples_text):
            for col1 in columns:
                (start1, end1) = df[col1].iloc[i] if df[col1].iloc[i] else (None, None)
                if start1:
                    for col2 in columns:
                        (start2, end2) = df[col2].iloc[i] if df[col2].iloc[i] else (None, None)
                        if col1 != col2 and start2:
                            if self.overlap(start1, end1, start2, end2):
                                print(f'conflict between {col1} and {col2}')
                                self.controls[col1].setStyleSheet("background-color: red")
                                self.controls[col2].setStyleSheet("background-color: red")
                                conflicting_columns.add(col1)
                                conflicting_columns.add(col2)

            df = pandas.DataFrame.from_dict(self.get_match_spans(matches, 2))
            for col in df.sort_values(0, axis=1, ascending=False):
                if col not in conflicting_columns:
                    r, g, b, a = self.colours[col].getRgb()
                    (start, end) = df[col].iloc[i] if df[col].iloc[i] else (None, None)
                    if start:
                        file_name = f'{file_name[:start]}<span style="background-color: rgb({r}, {g}, {b})">{file_name[start:end]}</span>{file_name[end:]}'
                    else:
                        self.controls[col].setStyleSheet("background-color: orange")
            try:
                self.samples[i].setText(file_name)
            except:
                ...
                # print(i, self.samples)

    def overlap(self, start1, end1, start2, end2):
        return ((start1 <= start2 < end1) or (start1 < end2 <= end1)
                or (start2 <= start1 < end2) or (start2 < end1 <= end2))

    def colourize_labels(self, pattern, colour):
        r, g, b, a = colour.getRgb()
        style_sheet = f'\g<1><span style="background-color: rgb({r}, {g}, {b})">\g<2></span>\g<3>'
        for i, label_text in enumerate(self.samples):
            self.samples[i].setText(re.sub(pattern, style_sheet, self.samples[i].text()))

    def clear_colours(self):
        for i, _ in enumerate(self.samples):
            self.samples[i].setText(self.samples_text[i])

    def choose_colour(self, object_name):
        target, _ = str.split(object_name, '_')
        colour_chooser = QColorDialog(self.colours[target], self)
        colour_chooser.exec()
        if colour_chooser.selectedColor().isValid():
            self.colours[target] = colour_chooser.selectedColor()
        self.show_chosen_colours()
        self.change_sample_style()

    def show_chosen_colours(self):
        colours = {
            'FOV': self.ui.pos_colour,
            'C': self.ui.c_colour,
            'T': self.ui.t_colour,
            'Z': self.ui.z_colour,
        }
        borders = {
            'FOV': self.ui.pos_pattern,
            'C': self.ui.c_pattern,
            'T': self.ui.t_pattern,
            'Z': self.ui.z_pattern,
        }
        for pattern, colour in self.colours.items():
            r, g, b, a = colour.getRgb()
            colours[pattern].setStyleSheet(f"background-color: rgb({r}, {g}, {b});")
            borders[pattern].setStyleSheet(f"border: 2px solid rgb({r}, {g}, {b});")

    def button_clicked(self, button):
        clicked_button = button.parent().standardButton(button)
        match clicked_button:
            case QDialogButtonBox.StandardButton.Ok:
                regexes = self.get_regex()
                df = pandas.DataFrame.from_dict(self.get_match_spans(self.find_matches(regexes), 0))
                regex = '.*'.join([regexes[col] for col in df.sort_values(0, axis=1, ascending=True).columns])
                wait_dialog = WaitDialog('Creating Fields of view', self, cancel_msg='Cancel FOV creation: please wait',
                                         progress_bar=True, )
                self.finished.connect(wait_dialog.close_window)
                self.progress.connect(wait_dialog.show_progress)
                wait_dialog.wait_for(self.create_fov_annotate, regex)
                # self.create_fov_annotate(regex)
                with pydetecdiv_project(PyDetecDiv().project_name) as project:
                    project.raw_dataset.pattern = regex
                    project.save(project.raw_dataset)
                self.close()
            case QDialogButtonBox.StandardButton.Cancel:
                self.close()
            case QDialogButtonBox.StandardButton.Reset:
                self.reset()

    def create_fov_annotate(self, regex):
        with pydetecdiv_project(PyDetecDiv().project_name) as project:
            pattern = re.compile(regex)
            fov_index = pattern.groupindex['FOV']
            fov_pattern = ''.join(re.findall(r'\(.*?\)', regex)[fov_index - 2:fov_index + 1])
            project.annotate(project.raw_dataset, 'url', tuple(pattern.groupindex.keys()), regex)
            for i in project.create_fov_from_raw_data('url', fov_pattern):
                self.progress.emit(i)
                if QThread.currentThread().isInterruptionRequested():
                    project.cancel()
                    break
        self.finished.emit(True)


