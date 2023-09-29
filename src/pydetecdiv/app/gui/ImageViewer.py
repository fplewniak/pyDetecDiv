import os
import time
import pandas as pd
from PySide6.QtCore import Signal, Qt, QRect, QPoint, QTimer
from PySide6.QtGui import QPixmap, QImage, QPen, QTransform, QKeySequence
from PySide6.QtWidgets import QMainWindow, QGraphicsScene, QGraphicsItem, QGraphicsRectItem, QFileDialog, QMenu
import numpy as np
import cv2 as cv
from skimage.feature import peak_local_max

from pydetecdiv.app import WaitDialog, PyDetecDiv, DrawingTools, pydetecdiv_project
from pydetecdiv.app.gui.ui.ImageViewer import Ui_ImageViewer
from pydetecdiv.domain.ImageResource import ImageResource
from pydetecdiv.domain.ROI import ROI
from pydetecdiv.settings import get_config_value
from pydetecdiv.utils import round_to_even


class ImageViewer(QMainWindow, Ui_ImageViewer):
    """
    Class to view and manipulate an Image resource
    """
    video_frame = Signal(int)
    finished = Signal(bool)

    def __init__(self, **kwargs):
        QMainWindow.__init__(self)
        self.ui = Ui_ImageViewer()
        self.setWindowTitle('Image viewer')
        self.ui.setupUi(self)
        self.scale = 100
        self.ui.z_slider.setEnabled(False)
        self.ui.t_slider.setEnabled(False)
        self.ui.z_slider.setPageStep(1)
        self.image_resource = None
        self.scene = ViewerScene()
        self.scene.setParent(self)
        self.pixmap = QPixmap()
        self.pixmapItem = self.scene.addPixmap(self.pixmap)
        self.ui.viewer.setScene(self.scene)
        self.fov = None
        self.stage = None
        self.project_name = None
        self.C = 0
        self.T = 0
        self.Z = 0
        self.parent_viewer = None
        self.image_source_ref = None
        # self.drift = None
        self.apply_drift = False
        self.roi_template = None
        self.video_playing = False
        self.video_frame.emit(self.T)
        self.video_frame.connect(lambda frame: self.ui.current_frame.setText(f'Frame: {frame}'))
        self.video_frame.connect(self.ui.t_slider.setSliderPosition)
        self.crop = None
        self.timer = None
        self.start = None
        self.first_frame = None
        self.frame = None
        self.wait = None

    def set_image_resource(self, image_resource, crop=None):
        """
        Associate an image resource to this viewer, possibly cropping if requested.

        :param image_resource: the image resource to load into the viewer
        :type image_resource: ImageResource
        :param crop: the (X,Y) crop area
        :type crop: list of slices [X, Y]
        """
        self.image_resource = image_resource
        self.T, self.C, self.Z = (0, 0, 0)

        self.ui.view_name.setText(f'View: {image_resource.fov.name}')
        self.crop = crop

        self.ui.z_slider.setMinimum(0)
        self.ui.z_slider.setMaximum(image_resource.sizeZ - 1)
        self.ui.z_slider.setEnabled(True)

        self.ui.t_slider.setMinimum(0)
        self.ui.t_slider.setMaximum(image_resource.sizeT - 1)
        self.ui.t_slider.setEnabled(True)

    def set_channel(self, C):
        """
        Sets the current channel
        TODO: allow specification of channel by name, this method should set the self.C field to the index corresponding
        TODO: to the requested name if the C argument is a str

        :param C: index of the current channel
        :type C: int
        """
        self.C = C

    def zoom_reset(self):
        """
        Reset the zoom to 1:1
        """
        self.ui.viewer.scale(100 / self.scale, 100 / self.scale)
        self.scale = 100
        self.ui.zoom_value.setSliderPosition(100)
        self.ui.scale_value.setText(f'Zoom: {self.scale}%')

    def zoom_fit(self):
        """
        Set the zoom value to fit the image in the viewer
        """
        self.ui.viewer.fitInView(self.pixmapItem, Qt.KeepAspectRatio)
        self.scale = int(100 * np.around(self.ui.viewer.transform().m11(), 2))
        self.ui.zoom_value.setSliderPosition(self.scale)
        self.ui.scale_value.setText(f'Zoom: {self.scale}%')

    def zoom_set_value(self, value):
        """
        Set the zoom to the specified value

        :param value: the zoom value (as %)
        :type value: float
        """
        self.ui.viewer.scale(value / self.scale, value / self.scale)
        self.scale = value
        self.ui.scale_value.setText(f'Zoom: {self.scale}%')

    def play_video(self):
        """
        Play the video with a maximum of an image every 50 ms (i.e. 20 FPS). Note that if loading a frame takes longer
        than 50 ms, then the frame rate may be lower.
        """
        self.timer = QTimer()
        self.timer.timeout.connect(self.show_next_frame)
        self.timer.setInterval(50)
        self.start = time.time()
        self.video_playing = True
        self.first_frame = self.T
        self.frame = self.T
        self.timer.start()

    def show_next_frame(self):
        """
        Show next frame when playing a video
        """
        self.frame += 1
        if self.frame >= self.image_resource.sizeT or not self.video_playing:
            self.timer.stop()
        else:
            end = time.time()
            speed = np.around((self.frame - self.first_frame) / (end - self.start), 1)
            self.ui.FPS.setText(f'FPS: {speed}')
            self.change_frame(self.frame)

    def pause_video(self):
        """
        Pause the video by setting the video_playing flag to False
        """
        self.video_playing = False

    def video_back(self):
        """
        Reset the current frame to the first one
        """
        self.change_frame(T=0)

    def change_layer(self, Z=0):
        """
        Set the layer to the specified value and refresh the display

        :param Z: the Z layer index
        :type Z: int
        """
        if self.Z != Z:
            self.Z = Z
            self.display()

    def change_frame(self, T=0):
        """
        Change the current frame to the specified time index and refresh the display

        :param T:
        """
        if self.T != T:
            self.T = T
            self.video_frame.emit(self.T)
            self.display()

    def display(self, C=None, T=None, Z=None):
        """
        Display the frame specified by the time, channel and layer indices.

        :param C: the channel index
        :type C: int
        :param T: the time index
        :type T: int
        :param Z: the layer index
        :type Z: int
        """
        C = self.C if C is None else C
        T = self.T if T is None else T
        Z = self.Z if Z is None else Z
        if self.apply_drift:
            idx = T if T < len(self.parent().parent().drift) else T - 1
            arr = self.image_resource.image(C=C, T=T, Z=Z, drift=self.parent().parent().drift.iloc[idx])
        else:
            arr = self.image_resource.image(C=C, T=T, Z=Z)
        # print(self.crop)
        if self.crop is not None:
            arr = arr[..., self.crop[1], self.crop[0]]
            # print(f'cropping to {self.crop}')
        ny, nx = arr.shape
        # print(f'display shape: {arr.shape}')
        img = QImage(np.ascontiguousarray(arr.data), nx, ny, QImage.Format_Grayscale16)
        self.pixmap.convertFromImage(img)
        self.pixmapItem.setPixmap(self.pixmap)

    def draw_saved_rois(self, roi_list):
        """
        Draw saved ROIs as green rectangles that can be selected but not moved

        :param roi_list: the list of saved ROIs
        :type roi_list: list of ROI objects
        """
        for roi in roi_list:
            rect_item = self.scene.addRect(QRect(0, 0, roi.width, roi.height))
            rect_item.setPen(self.scene.saved_pen)
            rect_item.setPos(QPoint(roi.x, roi.y))
            rect_item.setFlags(QGraphicsItem.ItemIsSelectable)
            rect_item.setData(0, roi.name)

    def close_window(self):
        """
        Close the Tabbed viewer containing this Image viewer
        """
        self.parent().parent().window.close()

    def compute_and_plot_drift(self, method='vidstab'):
        """
        Slot to compute the drift correction from the image resource displayed in this viewer, then plot the (x,y) drift
        against frame index. This slot runs compute_drift() with the requested method, launches a message dialog window
        and waits for completion before displaying the plot
        """
        self.wait = WaitDialog('Computing drift, please wait.', self, cancel_msg='Cancel drift computation please wait')
        self.finished.connect(self.wait.close_window)
        self.wait.wait_for(self.compute_drift, method=method)
        for viewer in self.parent().parent().get_image_viewers():
            viewer.ui.actionPlot.setEnabled(True)
            viewer.ui.actionApply_correction.setEnabled(True)
            viewer.ui.actionSave_to_file.setEnabled(True)
        self.plot_drift(method)

    def plot_drift(self, method):
        """
        Open a MatplotViewer tab and plot the (x,y) drift against frame index
        """
        self.parent().parent().show_plot(self.parent().parent().drift, f'Drift - {method}')

    def compute_drift(self, method='vidstab'):
        """
        Computation and update of the drift values. When the computation is over, this method emits a finished signal
        """
        self.parent().parent().drift = self.image_resource.compute_drift(Z=self.Z, C=self.C, method=method)
        self.finished.emit(True)

    def compute_drift_vidstab(self):
        """
        Computation and update of the drift values using the 'Vidstab' package method
        """
        self.compute_and_plot_drift()

    def compute_drift_phase_correlation(self):
        """
        Computation and update of the drift values using the 'phase correlation' method from OpenCV package
        """
        self.compute_and_plot_drift(method='phase correlation')

    def apply_drift_correction(self):
        """
        Apply the drift correction to the display and reload the image data with extra margins according to the drift
        values
        """
        self.apply_drift = self.ui.actionApply_correction.isChecked()
        if self.image_source_ref and self.parent_viewer:
            data, crop = self.parent_viewer.get_roi_data(self.image_source_ref)
            self.set_image_resource(ImageResource(data=data, fov=self.image_resource.fov), crop=crop)
        self.display()

    def load_drift_file(self):
        """
        Load a CSV file containing (x, y) drift values
        """
        drift_filename, _ = QFileDialog.getOpenFileName(
            dir=os.path.join(get_config_value('project', 'workspace'), PyDetecDiv().project_name), filter='*.csv')
        if os.path.isfile(drift_filename):
            self.parent().parent().drift = pd.read_csv(drift_filename)
            self.ui.actionPlot.setEnabled(True)
            self.ui.actionApply_correction.setEnabled(True)
            self.ui.actionSave_to_file.setEnabled(True)
        else:
            self.parent().parent().drift = None

    def save_drift_file(self):
        """
        Save (x, y) drift values to a file
        """
        drift_filename, _ = QFileDialog.getSaveFileName(
            dir=os.path.join(get_config_value('project', 'workspace'), PyDetecDiv().project_name), filter='*.csv')
        if drift_filename:
            self.parent().parent().drift.to_csv(drift_filename, index=False)

    def set_roi_template(self):
        """
        Set the currently selected area as a template to define other ROIs
        """
        roi = self.scene.get_selected_ROI()
        if roi:
            data = self.get_roi_image(roi)
            self.roi_template = np.uint8(np.array(data) / np.max(data) * 255)
            self.ui.actionIdentify_ROIs.setEnabled(True)
            self.ui.actionView_template.setEnabled(True)

    def get_roi_image(self, roi):
        """
        Get a 2D image of the specified ROI

        :param roi: the specified area to get image data for
        :type roi: QRect
        :return: the image data
        :rtype: 2D ndarray
        """
        w, h = roi.rect().toRect().size().toTuple()
        pos = roi.pos()
        x1, x2 = int(pos.x()), w + int(pos.x())
        y1, y2 = int(pos.y()), h + int(pos.y())
        return self.image_resource.image(C=self.C, T=self.T, Z=self.Z)[y1:y2, x1:x2]

    def get_roi_data(self, roi):
        """
        Get a 5D image resource for the specified ROI. If the drift correction is available, then the (x,y) crop is
        larger than the actual ROI to allow drift correction.

        :param roi: the specified area to get image resource data for
        :type roi: QRect
        :return: the image data
        :rtype: 5D ndarray
        """
        w, h = roi.rect().toRect().size().toTuple()
        pos = roi.pos()
        x1, x2 = int(pos.x()), w + int(pos.x())
        y1, y2 = int(pos.y()), h + int(pos.y())
        crop = None
        if self.parent().parent().drift is not None:
            max_shift_x = np.max(np.abs(self.parent().parent().drift.dx))
            max_shift_y = np.max(np.abs(self.parent().parent().drift.dy))
            x1, x2 = round_to_even(x1 - max_shift_x, ceil=False), round_to_even(x2 + max_shift_x)
            y1, y2 = round_to_even(y1 - max_shift_y, ceil=False), round_to_even(y2 + max_shift_y)
            x, y = round_to_even(max_shift_x, ceil=False), round_to_even(max_shift_y, ceil=False)
            crop = [slice(x, x + w), slice(y, y + h)]
            # min_shift_x, max_shift_x = np.min(self.drift.dx), np.max(self.drift.dx)
            # min_shift_y, max_shift_y = np.min(self.drift.dy), np.max(self.drift.dy)
            # print(min_shift_x, max_shift_x, min_shift_y, max_shift_y)
            # print(f'without margins: ({x1}, {y1}) - ({x2}, {y2}) [{x2 - x1}, {y2 - y1}]')
            # x1 = round_to_even(np.min([x1 + min_shift_x, x1]), ceil=False)
            # x2 = round_to_even(np.max([x2 + max_shift_x, x2]))
            # y1 = round_to_even(np.min([y1 + min_shift_y, y1]), ceil=False)
            # y2 = round_to_even(np.max([y2 + max_shift_y, y2]))
            # print(f'   with margins: ({x1}, {y1}) - ({x2}, {y2}) [{x2 - x1}, {y2 - y1}]')
            # x = round_to_even(max_shift_x, ceil=False) if max_shift_x > 0 else 0
            # y = round_to_even(max_shift_y, ceil=False) if max_shift_y > 0 else 0
            # crop = [slice(x, x+w), slice(y, y+h)]
            # print(f'should crop to {x}, {y}, {x+w}, {y+h} - {self.crop} - {w}, {h}')
        return self.image_resource.data_sample(X=slice(x1, x2), Y=slice(y1, y2)), crop

    def load_roi_template(self):
        """
        Load ROI template from a file.
        """
        filename = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.tif *.tiff)")[0]
        img = cv.imread(filename, cv.IMREAD_GRAYSCALE)
        self.roi_template = np.uint8(np.array(img / np.max(img) * 255))
        self.ui.actionIdentify_ROIs.setEnabled(True)

    def identify_rois(self):
        """
        Identify ROIs in an image using the ROI template as a model and the matchTemplate function from OpenCV
        """
        threshold = 0.3
        img = self.image_resource.image(C=self.C, Z=self.Z, T=self.T)
        img8bits = np.uint8(np.array(img / np.max(img) * 255))
        res = cv.matchTemplate(img8bits, self.roi_template, cv.TM_CCOEFF_NORMED)
        # loc = np.where(res >= threshold)
        # xy = peak_local_max(res,min_distance=self.roi_template.shape[0],threshold_abs=threshold,exclude_border=False)
        xy = peak_local_max(res, threshold_abs=threshold, exclude_border=False)
        w, h = self.roi_template.shape[::-1]
        for pt in xy:
            x, y = pt[1], pt[0]
            if not isinstance(self.scene.itemAt(QPoint(x, y), QTransform().scale(1, 1)), QGraphicsRectItem):
                rect_item = self.scene.addRect(QRect(0, 0, w, h))
                rect_item.setPos(x, y)
                if [r for r in rect_item.collidingItems(Qt.IntersectsItemBoundingRect) if
                    isinstance(r, QGraphicsRectItem)]:
                    self.scene.removeItem(rect_item)
                else:
                    rect_item.setPen(self.scene.match_pen)
                    rect_item.setFlags(QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemIsSelectable)

    def view_template(self):
        """
        Display the currently selected template in a new tab as a 2D image.
        """
        self.parent().parent().show_image(self.roi_template, title='ROI template', format_=QImage.Format_Grayscale8)

    def view_roi_image(self, selected_roi=None):
        """
        Display the selected area in a new tab viewer as a 5D image resource.

        :param selected_roi:
        """
        viewer = ImageViewer()
        viewer.image_source_ref = selected_roi if selected_roi else self.scene.get_selected_ROI()
        viewer.parent_viewer = self
        data, crop = self.get_roi_data(viewer.image_source_ref)
        self.parent().parent().addTab(viewer, viewer.image_source_ref.data(0))
        viewer.set_image_resource(ImageResource(data=data, fov=self.image_resource.fov), crop=crop)
        viewer.ui.view_name.setText(f'View: {viewer.image_source_ref.data(0)}')
        viewer.display()
        self.parent().parent().setCurrentWidget(viewer)

    def save_rois(self):
        """
        Save the areas as ROIs
        """
        rois = [item for item in self.scene.items() if isinstance(item, QGraphicsRectItem)]
        with pydetecdiv_project(PyDetecDiv().project_name) as project:
            for i, rect_item in enumerate(sorted(rois, key=lambda x: x.scenePos().toPoint().toTuple())):
                x, y = rect_item.scenePos().toPoint().toTuple()
                w, h = rect_item.rect().toRect().getCoords()[2:]
                new_roi = ROI(project=project, name=f'{self.image_resource.fov.name}_{i}',
                              fov=self.image_resource.fov, top_left=(x, y), bottom_right=(int(x) + w, int(y) + h))
                rect_item.setData(0, new_roi.name)
        self.fixate_saved_rois()

    def fixate_saved_rois(self):
        """
        Disable the possibility to move ROIs once they have been saved
        """
        for r in [item for item in self.scene.items() if isinstance(item, QGraphicsRectItem)]:
            r.setPen(self.scene.saved_pen)
            r.setFlag(QGraphicsItem.ItemIsMovable, False)


class ViewerScene(QGraphicsScene):
    """
    The viewer scene where images and other items are drawn
    """

    def __init__(self):
        super().__init__()
        self.from_x = None
        self.from_y = None
        self.pen = QPen(Qt.GlobalColor.cyan, 2)
        self.match_pen = QPen(Qt.GlobalColor.yellow, 2)
        self.saved_pen = QPen(Qt.GlobalColor.green, 2)
        self.warning_pen = QPen(Qt.GlobalColor.red, 2)

    def contextMenuEvent(self, event):
        """
        The context menu for area manipulation

        :param event:
        """
        menu = QMenu()
        view_in_new_tab = menu.addAction("View in new tab")
        r = self.itemAt(event.scenePos(), QTransform().scale(1, 1))
        if isinstance(r, QGraphicsRectItem):
            view_in_new_tab.triggered.connect(lambda _: self.parent().view_roi_image(r))
        menu.exec(event.screenPos())

    def keyPressEvent(self, event):
        """
        Detect when a key is pressed and perform the corresponding action:
        * QKeySequence.Delete: delete the selected item

        :param event: the key pressed event
        :type event: QKeyEvent
        """
        if event.matches(QKeySequence.Delete):
            for r in self.selectedItems():
                self.removeItem(r)

    def mousePressEvent(self, event):
        """
        Detect when the left mouse button is pressed and perform the action corresponding to the currently checked
        drawing tool

        :param event: the mouse press event
        :type event: QGraphicsSceneMouseEvent
        """
        if event.button() == Qt.LeftButton:
            match PyDetecDiv().current_drawing_tool:
                case DrawingTools.Cursor:
                    self.select_ROI(event)
                case DrawingTools.DrawROI:
                    self.select_ROI(event)
                case DrawingTools.DuplicateROI:
                    self.duplicate_selected_ROI(event)

    def select_ROI(self, event):
        """
        Select the current area/ROI

        :param event: the mouse press event
        :type event: QGraphicsSceneMouseEvent
        """
        _ = [r.setSelected(False) for r in self.items()]
        r = self.itemAt(event.scenePos(), QTransform().scale(1, 1))
        if isinstance(r, QGraphicsRectItem):
            r.setSelected(True)
        if self.selectedItems():
            self.parent().ui.actionSet_template.setEnabled(True)
        else:
            self.parent().ui.actionSet_template.setEnabled(False)

    def get_selected_ROI(self):
        """
        Return the selected ROI

        :return: the selected ROI
        :rtype: QGraphicsRectItem
        """
        for selection in self.selectedItems():
            if isinstance(selection, QGraphicsRectItem):
                return selection
        return None

    def duplicate_selected_ROI(self, event):
        """
        Duplicate the currently selected ROI at the current mouse position

        :param event: the mouse press event
        :type event: QGraphicsSceneMouseEvent
        """
        pos = event.scenePos()
        roi = self.get_selected_ROI()
        if roi:
            roi = self.addRect(roi.rect())
            roi.setPen(self.pen)
            w, h = roi.rect().size().toTuple()
            roi.setPos(QPoint(pos.x() - np.around(w / 2.0), pos.y() - np.around(h / 2.0)))
            roi.setFlags(QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemIsSelectable)
            roi.setData(0, f'Region{len(self.items())}')
            self.select_ROI(event)
            if [r for r in roi.collidingItems(Qt.IntersectsItemBoundingRect) if isinstance(r, QGraphicsRectItem)]:
                roi.setPen(self.warning_pen)
            else:
                roi.setPen(self.pen)

    def mouseMoveEvent(self, event):
        """
        Detect mouse movement and apply the appropriate method according to the currently cjecked drawing tool and key
        modifier

        :param event: the mouse move event
        :type event: QGraphicsSceneMouseEvent
        """
        if event.button() == Qt.NoButton:
            match PyDetecDiv().current_drawing_tool, event.modifiers():
                case DrawingTools.Cursor, Qt.NoModifier:
                    self.move_ROI(event)
                case DrawingTools.Cursor, Qt.ControlModifier:
                    self.draw_ROI(event)
                case DrawingTools.DrawROI, Qt.NoModifier:
                    self.draw_ROI(event)
                case DrawingTools.DrawROI, Qt.ControlModifier:
                    self.move_ROI(event)
                case DrawingTools.DuplicateROI, Qt.NoModifier:
                    self.move_ROI(event)

    def move_ROI(self, event):
        """
        Move the currently selected ROI if it is movable

        :param event: the mouse move event
        :type event: QGraphicsSceneMouseEvent
        """
        roi = self.get_selected_ROI()
        if roi and (roi.flags() & QGraphicsItem.ItemIsMovable):
            pos = event.scenePos()
            roi.moveBy(pos.x() - event.lastScenePos().x(), pos.y() - event.lastScenePos().y())
            if [r for r in roi.collidingItems(Qt.IntersectsItemBoundingRect) if isinstance(r, QGraphicsRectItem)]:
                roi.setPen(self.warning_pen)
            else:
                roi.setPen(self.pen)

    def draw_ROI(self, event):
        """
        Draw or redraw the currently selected ROI if it is movable

        :param event: the mouse press event
        :type event: QGraphicsSceneMouseEvent
        """
        roi = self.get_selected_ROI()
        pos = event.scenePos()
        if roi and (roi.flags() & QGraphicsItem.ItemIsMovable):
            roi_pos = roi.scenePos()
            w, h = round_to_even(pos.x() - roi_pos.x()), round_to_even(pos.y() - roi_pos.y())
            rect = QRect(0, 0, w, h)
            roi.setRect(rect)
        else:
            roi = self.addRect(QRect(0, 0, 2, 2))
            roi.setPen(self.pen)
            roi.setPos(QPoint(pos.x(), pos.y()))
            roi.setFlags(QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemIsSelectable)
            roi.setData(0, f'Region{len(self.items())}')
            self.select_ROI(event)
        if [r for r in roi.collidingItems(Qt.IntersectsItemBoundingRect) if isinstance(r, QGraphicsRectItem)]:
            roi.setPen(self.warning_pen)
        else:
            roi.setPen(self.pen)
