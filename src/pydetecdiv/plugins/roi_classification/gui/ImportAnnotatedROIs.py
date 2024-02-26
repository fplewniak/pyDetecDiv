"""
Dialog window handling the definition of patterns for FOV creation from raw data file names
"""
import random
import re

import numpy as np
import pandas
import pandas as pd
from PySide6.QtCore import Signal, QThread
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QDialog, QColorDialog, QDialogButtonBox

from pydetecdiv.domain.ROI import ROI
from pydetecdiv.plugins.roi_classification.gui.ui.ImportAnnotatedROIs import Ui_FOV2ROIlinks
from pydetecdiv.app import PyDetecDiv, pydetecdiv_project, WaitDialog
from pydetecdiv.settings import get_config_value

class FOV2ROIlinks(QDialog, Ui_FOV2ROIlinks):
    """
    A class extending the QDialog and the Ui_RawData2FOV classes. Ui_RawData2FOV was created using QTDesigner
    """
    finished = Signal(bool)
    progress = Signal(int)

    def __init__(self, annotation_file, plugin):
        # Base class
        QDialog.__init__(self, PyDetecDiv().main_window)

        # Initialize the UI widgets
        self.ui = Ui_FOV2ROIlinks()
        self.ui.setupUi(self)

        self.plugin = plugin
        self.annotation_file = annotation_file

        self.colours = {
            'FOV': QColor.fromRgb(0, 200, 0, 255)
        }
        with pydetecdiv_project(PyDetecDiv().project_name) as project:
            fov_names = [fov.name for fov in project.get_objects('FOV')]
            self.FOVsamples_text = random.sample(fov_names, min([len(fov_names), 5]))

        self.FOVsamples = [self.ui.sampleFOV1, self.ui.sampleFOV2,
                           self.ui.sampleFOV3, self.ui.sampleFOV4,
                           self.ui.sampleFOV5][
                          :min([len(fov_names), 5])]
        for i, label_text in enumerate(self.FOVsamples_text):
            self.FOVsamples[i].setText(label_text)

        self.df = pd.read_csv(annotation_file)
        self.df['frame'] -= 1
        # self.class_index_mapping =  [-1] * len(self.plugin.class_names)
        # self.class_index_mapping = {row.ann: self.plugin.class_names.index(row.class_name)
        #                             for row in self.df.groupby(['ann', 'class_name']).size().
        #                             reset_index(name='count').itertuples()}
        # self.df['ann'] = self.df['ann'].replace(self.class_index_mapping)
        # self.df['ann'] -= 1
        self.ROIsamples = self.df['roi'].tolist()
        self.ROIsamples_text = random.sample(self.ROIsamples, min([len(self.df), 5]))
        self.ROIsamples = [self.ui.sampleROI1, self.ui.sampleROI2,
                           self.ui.sampleROI3, self.ui.sampleROI4,
                           self.ui.sampleROI5][
                          :min([len(self.df), 5])]

        self.controls = {'FOV': self.ui.position,
                         }
        for i, label_text in enumerate(self.ROIsamples_text):
            self.ROIsamples[i].setText(label_text)
        self.setWindowTitle('Create links between annotated ROIs and FOVs')
        self.ui.pos_left.addItems(['position', 'Pos', ''])
        self.ui.pos_pattern.addItems(['\d+', 'position\d+', 'Pos\d+'])
        self.reset()
        self.exec()
        for child in self.children():
            child.deleteLater()
        self.destroy(True)

    def reset(self):
        """
        Reset the form with default patterns and colours
        """
        self.colours = {
            'FOV': QColor.fromRgb(0, 200, 0, 255)
        }
        self.ui.pos_left.setCurrentIndex(0)
        self.ui.pos_pattern.setCurrentIndex(0)
        self.ui.pos_right.setCurrentText('')

        self.show_chosen_colours()
        self.change_sample_style()

    def get_regex(self):
        """
        Build the complete regular expression from the individual patterns

        :return: the regular expression string
        """
        regex = {}
        patterns = [self.ui.pos_left.currentText(),
                    self.ui.pos_pattern.currentText(),
                    self.ui.pos_right.currentText()]
        for i, _ in enumerate(patterns):
            while patterns[i].endswith('\\'):
                patterns[i] = patterns[i][:-1]
        regex['FOV'] = f'({patterns[0]})(?P<FOV>{patterns[1]})({patterns[2]})'
        return regex

    def change_sample_style(self):
        """
        Change the colours of file name samples showing the pattern matches.
        """
        self.clear_colours()
        regex = self.get_regex()
        if regex:
            self.colourize_matches(self.find_matches(self.ROIsamples_text, regex))

    def find_matches(self, fov_names, regexes):
        """
        Find a list of matches with the defined regular expressions

        :param regexes: the list of regular expressions to match
        :return: a list of matches
        """
        matches = {}
        for what in regexes:
            pattern = re.compile(regexes[what])
            matches[what] = [re.search(pattern, label_text) for label_text in fov_names]
        return matches

    @staticmethod
    def get_match_spans(matches, group):
        """
        Get the list of group positions for matches

        :param matches: the list of matches
        :param group: the group index to retrieve the spans for
        :return: a dictionary of spans for the patterns (FOV, C, T, Z)
        """
        return {what: [FOV2ROIlinks.get_match_span(match, group) for match in matches[what]] for what in matches}

    @staticmethod
    def get_match_span(match, group=2):
        """
        Get the span of a given group in a match

        :param match: the match
        :param group: the group index
        :return: the group match span
        """
        if match:
            return match.span(group)
        return None

    def colourize_matches(self, matches):
        """
        Find matches in file name samples and colourize them accordingly. Non-matching pattern check boxes' background
        is set to orange. Conflicting patterns (having overlapping matches) are shown in red.

        :param matches: the list of matches to colourize
        """
        df = pandas.DataFrame.from_dict(self.get_match_spans(matches, 0))
        columns = set(df.columns)
        conflicting_columns = set()
        self.ui.buttonBox.button(QDialogButtonBox.Ok).setEnabled(True)
        for col in columns:
            self.controls[col].setStyleSheet("")
        for i, roi_name in enumerate(self.ROIsamples_text):
            # for col1 in columns:
            #     (start1, end1) = df[col1].iloc[i] if df[col1].iloc[i] else (None, None)
            #     print(start1, end1)
            #     if start1:
            #         for col2 in columns:
            #             (start2, end2) = df[col2].iloc[i] if df[col2].iloc[i] else (None, None)
            #             if col1 != col2 and start2:
            #                 if self.overlap(start1, end1, start2, end2):
            #                     print(f'conflict between {col1} and {col2}')
            #                     self.controls[col1].setStyleSheet("background-color: red")
            #                     self.controls[col2].setStyleSheet("background-color: red")
            #                     conflicting_columns.add(col1)
            #                     conflicting_columns.add(col2)
            #                     self.ui.buttonBox.button(QDialogButtonBox.Ok).setEnabled(False)
            df = pandas.DataFrame.from_dict(self.get_match_spans(matches, 2))
            for col in df.sort_values(0, axis=1, ascending=False):
                if col not in conflicting_columns:
                    r, g, b, _ = self.colours[col].getRgb()
                    (start, end) = df[col].iloc[i] if df[col].iloc[i] else (None, None)
                    if start is not None:
                        roi_name = f'{roi_name[:start]}<span style="background-color: rgb({r}, {g}, {b})">{roi_name[start:end]}</span>{roi_name[end:]}'
                    else:
                        self.controls[col].setStyleSheet("background-color: orange")
                        self.ui.buttonBox.button(QDialogButtonBox.Ok).setEnabled(False)
            try:
                self.ROIsamples[i].setText(roi_name)
            finally:
                ...
                # print(i, self.ROIsamples)

    def overlap(self, start1, end1, start2, end2):
        """
        Checks whether positions overlap

        :param start1: start of first span
        :param end1: end of first span
        :param start2: start of second span
        :param end2: end of second span
        :return: True if spans overlap, False otherwise
        """
        return ((start1 <= start2 < end1) or (start1 < end2 <= end1)
                or (start2 <= start1 < end2) or (start2 < end1 <= end2))

    def colourize_labels(self, pattern, colour):
        """
        Colourize the file name samples matching the pattern with the specified colour

        :param pattern: the pattern
        :param colour: the colour
        """
        r, g, b, _ = colour.getRgb()
        style_sheet = rf'\g<1><span style="background-color: rgb({r}, {g}, {b})">\g<2></span>\g<3>'
        for i, _ in enumerate(self.ROIsamples):
            self.ROIsamples[i].setText(re.sub(pattern, style_sheet, self.ROIsamples[i].text()))

    def clear_colours(self):
        """
        Clear colours to avoid overlapping style sheets
        """
        for i, _ in enumerate(self.ROIsamples):
            self.ROIsamples[i].setText(self.ROIsamples_text[i])

    def choose_colour(self, object_name):
        """
        Choose colour for a given pattern specified by its object name

        :param object_name: the object name
        """
        target, _ = str.split(object_name, '_')
        colour_chooser = QColorDialog(self.colours[target], self)
        colour_chooser.exec()
        if colour_chooser.selectedColor().isValid():
            self.colours[target] = colour_chooser.selectedColor()
        self.show_chosen_colours()
        self.change_sample_style()

    def show_chosen_colours(self):
        """
        Show the chosen colour in the little square box on the right and the border of the pattern.
        """
        colours = {
            'FOV': self.ui.pos_colour
        }
        borders = {
            'FOV': self.ui.pos_pattern
        }
        for pattern, colour in self.colours.items():
            r, g, b, _ = colour.getRgb()
            colours[pattern].setStyleSheet(f"background-color: rgb({r}, {g}, {b});")
            borders[pattern].setStyleSheet(f"border: 2px solid rgb({r}, {g}, {b});")

    def button_clicked(self, button):
        """
        React to clicked button

        :param button: the button that was clicked
        """
        clicked_button = button.parent().standardButton(button)
        match clicked_button:
            case QDialogButtonBox.StandardButton.Ok:
                pass
                regexes = self.get_regex()
                df = pandas.DataFrame.from_dict(
                    self.get_match_spans(self.find_matches(self.ROIsamples_text, regexes), 0))
                regex = '.*'.join([regexes[col] for col in df.sort_values(0, axis=1, ascending=True).columns])
                wait_dialog = WaitDialog('Importing annotated ROIs', self,
                                         cancel_msg='Cancel ROI creation: please wait',
                                         progress_bar=True, )
                self.finished.connect(wait_dialog.close_window)
                self.progress.connect(wait_dialog.show_progress)
                wait_dialog.wait_for(self.create_annotated_rois, regex)
                PyDetecDiv().project_selected.emit(PyDetecDiv().project_name)
                self.close()
            case QDialogButtonBox.StandardButton.Close:
                self.close()
            case QDialogButtonBox.StandardButton.Reset:
                self.reset()

    def create_annotated_rois(self, regex):
        """
        The actual FOV creation and data annotation method

        :param regex: the regular expression to use for data annotation
        """
        with pydetecdiv_project(PyDetecDiv().project_name) as project:
            run = self.plugin.save_run(project, 'import_annotated_rois',
                                            {'class_names': self.plugin.class_names,
                                             'annotator': get_config_value('project', 'user'),
                                             'file_name': self.annotation_file
                                             })

            fov_list = {f.name: f for f in project.get_objects('FOV')}
            new_roi_list = {}
            for row in self.df.groupby(['roi', 'x', 'y', 'width', 'height']).size().reset_index(
                    name='count').itertuples():
                match = re.search(regex, row.roi)
                new_roi_list[row.roi] = (ROI(project=project, name=row.roi, fov=fov_list[(match.group('FOV'))],
                                             top_left=(row.x, row.y),
                                             bottom_right=(row.x + row.width, row.y + row.height)))
                self.progress.emit(100.0 * row.Index / len(self.df))
                if QThread.currentThread().isInterruptionRequested():
                    project.cancel()
                    break

            for row in self.df.itertuples():
                self.plugin.save_results(project, run, new_roi_list[row.roi], row.frame, row.class_name)
                # print(row.roi, row.frame, row.ann, row.class_name)
                self.progress.emit(100.0 * row.Index / len(self.df))
                if QThread.currentThread().isInterruptionRequested():
                    project.cancel()
                    break

            # columns = tuple(re.compile(regex).groupindex.keys())
            # for i in project.create_fov_from_raw_data(project.annotate(project.raw_dataset, 'url', columns, regex), multi=self.ui.multiple_files.isChecked()):
            #     self.progress.emit(i)

        self.finished.emit(True)
