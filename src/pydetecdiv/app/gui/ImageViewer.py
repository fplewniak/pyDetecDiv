# """
# Image viewer to display and interact with an Image resource (5D image data)
# """
#
# import time
# from PySide6.QtCore import Signal, Qt, QRect, QPoint, QTimer
# from PySide6.QtGui import QPixmap, QImage, QPen, QTransform, QKeySequence, QCursor
# from PySide6.QtWidgets import QMainWindow, QGraphicsScene, QGraphicsItem, QGraphicsRectItem, QFileDialog, QMenu
# import numpy as np
# import cv2 as cv
# from skimage.feature import peak_local_max
# import qimage2ndarray
#
# from pydetecdiv.app import PyDetecDiv, DrawingTools, pydetecdiv_project
# from pydetecdiv.app.gui.ui.ImageViewer import Ui_ImageViewer
# from pydetecdiv.domain import ROI
# from pydetecdiv.utils import round_to_even
#
#
# class ImageViewer(QMainWindow, Ui_ImageViewer):
#     """
#     Class to view and manipulate an Image resource
#     """
#     video_frame = Signal(int)
#     video_channel = Signal(int)
#     video_layer = Signal(int)
#     finished = Signal(bool)
#
#     def __init__(self, **kwargs):
#         QMainWindow.__init__(self)
#         self.ui = Ui_ImageViewer()
#         self.setWindowTitle('Image viewer')
#         self.ui.setupUi(self)
#         self.scale = 100
#         self.ui.z_slider.setEnabled(False)
#         self.ui.t_slider.setEnabled(False)
#         self.ui.c_slider.setEnabled(False)
#         self.ui.z_slider.setPageStep(1)
#         self.ui.c_slider.setPageStep(1)
#         self.image_resource_data = None
#         self.scene = ViewerScene()
#         self.scene.setParent(self)
#         self.pixmap = QPixmap()
#         self.pixmapItem = self.scene.addPixmap(self.pixmap)
#         self.ui.viewer.setScene(self.scene)
#         self.fov = None
#         self.stage = None
#         # self.project_name = None
#         self.C = 0
#         self.T = 0
#         self.Z = 0
#         self.parent_viewer = None
#         self.image_source_ref = None
#         # self.drift = None
#         self.apply_drift = False
#         self.roi_template = None
#         self.video_playing = False
#         self.video_frame.emit(self.T)
#         self.video_frame.connect(lambda frame: self.ui.current_frame.setText(f'Frame: {frame}'))
#         self.video_frame.connect(self.ui.t_slider.setSliderPosition)
#         self.video_channel.connect(self.ui.c_slider.setSliderPosition)
#         self.video_layer.connect(self.ui.z_slider.setSliderPosition)
#         self.crop = None
#         self.timer = None
#         self.start = None
#         self.first_frame = None
#         self.frame = None
#         self.wait = None
#
#     def set_image_resource_data(self, image_resource, crop=None, T=0, C=0, Z=0):
#         """
#         Associate an image resource to this viewer, possibly cropping if requested.
#
#         :param image_resource: the image resource to load into the viewer
#         :type image_resource: ImageResourceData
#         :param crop: the (X,Y) crop area
#         :type crop: list of slices [X, Y]
#         """
#         self.image_resource_data = image_resource
#         self.T, self.C, self.Z = (T, C, Z)
#
#         self.ui.view_name.setText(f'View: {image_resource.fov.name}')
#         self.crop = crop
#
#         self.ui.c_slider.setMinimum(0)
#         self.ui.c_slider.setMaximum(image_resource.sizeC - 1)
#         self.ui.c_slider.setEnabled(True)
#
#         self.ui.z_slider.setMinimum(0)
#         self.ui.z_slider.setMaximum(image_resource.sizeZ - 1)
#         self.ui.z_slider.setEnabled(True)
#
#         self.ui.t_slider.setMinimum(0)
#         self.ui.t_slider.setMaximum(image_resource.sizeT - 1)
#         self.ui.t_slider.setEnabled(True)
#
#     def set_channel(self, C):
#         """
#         Sets the current channel
#         TODO: allow specification of channel by name, this method should set the self.C field to the index corresponding
#         TODO: to the requested name if the C argument is a str
#
#         :param C: index of the current channel
#         :type C: int
#         """
#         self.C = C
#
#     def zoom_reset(self):
#         """
#         Reset the zoom to 1:1
#         """
#         self.ui.viewer.scale(100 / self.scale, 100 / self.scale)
#         self.scale = 100
#         self.ui.zoom_value.setSliderPosition(100)
#         self.ui.scale_value.setText(f'Zoom: {self.scale}%')
#
#     def zoom_fit(self):
#         """
#         Set the zoom value to fit the image in the viewer
#         """
#         self.ui.viewer.fitInView(self.pixmapItem, Qt.KeepAspectRatio)
#         self.scale = int(100 * np.around(self.ui.viewer.transform().m11(), 2))
#         self.ui.zoom_value.setSliderPosition(self.scale)
#         self.ui.scale_value.setText(f'Zoom: {self.scale}%')
#
#     def zoom_set_value(self, value):
#         """
#         Set the zoom to the specified value
#
#         :param value: the zoom value (as %)
#         :type value: float
#         """
#         self.ui.viewer.scale(value / self.scale, value / self.scale)
#         self.scale = value
#         self.ui.zoom_value.setValue(value)
#         self.ui.scale_value.setText(f'Zoom: {self.scale}%')
#
#     def play_video(self):
#         """
#         Play the video with a maximum of an image every 50 ms (i.e. 20 FPS). Note that if loading a frame takes longer
#         than 50 ms, then the frame rate may be lower.
#         """
#         self.timer = QTimer()
#         self.timer.timeout.connect(self.show_next_frame)
#         self.timer.setInterval(50)
#         self.start = time.time()
#         self.video_playing = True
#         self.first_frame = self.T
#         self.frame = self.T
#         self.timer.start()
#
#     def show_next_frame(self):
#         """
#         Show next frame when playing a video
#         """
#         self.frame += self.ui.t_step.value()
#         if self.frame >= self.image_resource_data.sizeT or not self.video_playing:
#             self.timer.stop()
#         else:
#             end = time.time()
#             speed = np.around((self.frame - self.first_frame) / ((end - self.start) * self.ui.t_step.value()), 1)
#             self.ui.FPS.setText(f'FPS: {speed}')
#             self.change_frame(self.frame)
#
#     def pause_video(self):
#         """
#         Pause the video by setting the video_playing flag to False
#         """
#         self.video_playing = False
#
#     def video_back(self):
#         """
#         Reset the current frame to the first one
#         """
#         self.change_frame(T=0)
#
#     def change_layer(self, Z=0):
#         """
#         Set the layer to the specified value and refresh the display
#
#         :param Z: the Z layer index
#         :type Z: int
#         """
#         if self.Z != Z:
#             self.Z = Z
#             self.display()
#
#     def change_frame(self, T=0):
#         """
#         Change the current frame to the specified time index and refresh the display
#
#         :param T:
#         """
#         if self.T != T:
#             self.T = T
#             self.video_frame.emit(self.T)
#             self.display()
#
#     def change_channel(self, C=0):
#         """
#         Change the current frame to the specified time index and refresh the display
#
#         :param T:
#         """
#         if self.C != C:
#             self.C = C
#             self.display()
#
#     def synchronize_with(self, other):
#         """
#         Synchronize the current viewer with another one. This is used to view a portion of an image in a new viewer
#         with the same T, C, Z coordinates as the original.
#         :param other: the other viewer to synchronize with
#         """
#         if self.Z != other.Z:
#             self.Z = other.Z
#             self.video_layer.emit(self.Z)
#         if self.C != other.C:
#             self.C = other.C
#             self.video_channel.emit(self.C)
#         if self.T != other.T:
#             self.T = other.T
#             self.video_frame.emit(self.T)
#         self.display()
#
#     def display(self, C=None, T=None, Z=None):
#         """
#         Display the frame specified by the time, channel and layer indices.
#
#         :param C: the channel index
#         :type C: int
#         :param T: the time index
#         :type T: int
#         :param Z: the layer index
#         :type Z: int
#         """
#         C = self.C if C is None else C
#         T = self.T if T is None else T
#         Z = self.Z if Z is None else Z
#
#         arr = self.image_resource_data.image(C=C, T=T, Z=Z, drift=PyDetecDiv.app.apply_drift)
#         if arr is not None:
#             if self.crop is not None:
#                 arr = arr[..., self.crop[1], self.crop[0]]
#
#             img = qimage2ndarray.array2qimage((arr / 257).astype(np.uint8))
#             self.pixmap.convertFromImage(img)
#             self.pixmapItem.setPixmap(self.pixmap)
#
#     def draw_saved_rois(self, roi_list):
#         """
#         Draw saved ROIs as green rectangles that can be selected but not moved
#
#         :param roi_list: the list of saved ROIs
#         :type roi_list: list of ROI objects
#         """
#         for roi in roi_list:
#             rect_item = self.scene.addRect(QRect(0, 0, roi.width, roi.height))
#             rect_item.setPen(self.scene.saved_pen)
#             rect_item.setPos(QPoint(roi.x, roi.y))
#             rect_item.setFlags(QGraphicsItem.ItemIsSelectable)
#             rect_item.setData(0, roi.name)
#
#     def close_window(self):
#         """
#         Close the Tabbed viewer containing this Image viewer
#         """
#         self.parent().parent().window.close()
#
#
#     def set_roi_template(self):
#         """
#         Set the currently selected area as a template to define other ROIs
#         """
#         roi = self.scene.get_selected_ROI()
#         if roi:
#             data = self.get_roi_image(roi)
#             self.roi_template = np.uint8(np.array(data) / np.max(data) * 255)
#             self.ui.actionIdentify_ROIs.setEnabled(True)
#             self.ui.actionView_template.setEnabled(True)
#
#     def get_roi_image(self, roi):
#         """
#         Get a 2D image of the specified ROI
#
#         :param roi: the specified area to get image data for
#         :type roi: QRect
#         :return: the image data
#         :rtype: 2D ndarray
#         """
#         w, h = roi.rect().toRect().size().toTuple()
#         pos = roi.pos()
#         x1, x2 = int(pos.x()), w + int(pos.x())
#         y1, y2 = int(pos.y()), h + int(pos.y())
#         return self.image_resource_data.image(sliceX=slice(x1, x2), sliceY=slice(y1, y2), C=self.C, T=self.T, Z=self.Z)
#
#     def get_roi_data(self, roi):
#         """
#         Get a 5D image resource for the specified ROI. If the drift correction is available, then the (x,y) crop is
#         larger than the actual ROI to allow drift correction.
#
#         :param roi: the specified area to get image resource data for
#         :type roi: QRect
#         :return: the image data
#         :rtype: 5D ndarray
#         """
#         w, h = roi.rect().toRect().size().toTuple()
#         pos = roi.pos()
#         x1, x2 = int(pos.x()), w + int(pos.x())
#         y1, y2 = int(pos.y()), h + int(pos.y())
#         crop = (slice(x1, x2), slice(y1, y2))
#         return self.image_resource_data, crop
#
#     def load_roi_template(self):
#         """
#         Load ROI template from a file.
#         """
#         filename = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.tif *.tiff)")[0]
#         img = cv.imread(filename, cv.IMREAD_GRAYSCALE)
#         self.roi_template = np.uint8(np.array(img / np.max(img) * 255))
#         self.ui.actionIdentify_ROIs.setEnabled(True)
#
#     def identify_rois(self):
#         """
#         Identify ROIs in an image using the ROI template as a model and the matchTemplate function from OpenCV
#         """
#         threshold = 0.3
#         img = self.image_resource_data.image(C=self.C, Z=self.Z, T=self.T)
#         img8bits = np.uint8(np.array(img / np.max(img) * 255))
#         res = cv.matchTemplate(img8bits, self.roi_template, cv.TM_CCOEFF_NORMED)
#         # loc = np.where(res >= threshold)
#         # xy = peak_local_max(res,min_distance=self.roi_template.shape[0],threshold_abs=threshold,exclude_border=False)
#         xy = peak_local_max(res, threshold_abs=threshold, exclude_border=False)
#         w, h = self.roi_template.shape[::-1]
#         for pt in xy:
#             x, y = pt[1], pt[0]
#             if not isinstance(self.scene.itemAt(QPoint(x, y), QTransform().scale(1, 1)), QGraphicsRectItem):
#                 rect_item = self.scene.addRect(QRect(0, 0, w, h))
#                 rect_item.setPos(x, y)
#                 if [r for r in rect_item.collidingItems(Qt.IntersectsItemBoundingRect) if
#                     isinstance(r, QGraphicsRectItem)]:
#                     self.scene.removeItem(rect_item)
#                 else:
#                     rect_item.setPen(self.scene.match_pen)
#                     rect_item.setFlags(QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemIsSelectable)
#
#     def view_template(self):
#         """
#         Display the currently selected template in a new tab as a 2D image.
#         """
#         self.parent().parent().show_image(self.roi_template, title='ROI template', format_=QImage.Format_Grayscale8)
#
#     def view_roi_image(self, selected_roi=None):
#         """
#         Display the selected area in a new tab viewer
#
#         :param selected_roi:
#         """
#         viewer = ImageViewer()
#         PyDetecDiv.app.setOverrideCursor(QCursor(Qt.WaitCursor))
#         viewer.image_source_ref = selected_roi if selected_roi else self.scene.get_selected_ROI()
#         viewer.parent_viewer = self
#         data, crop = self.get_roi_data(viewer.image_source_ref)
#         self.parent().parent().addTab(viewer, viewer.image_source_ref.data(0))
#         # viewer.set_image_resource_data(ArrayImageResource(data=data, fov=self.image_resource_data.fov, image_resource=self.image_resource_data.image_resource), crop=crop)
#         viewer.set_image_resource_data(self.image_resource_data, crop=crop)
#         viewer.ui.view_name.setText(f'View: {viewer.image_source_ref.data(0)}')
#         viewer.synchronize_with(self)
#         # viewer.display()
#         self.parent().parent().setCurrentWidget(viewer)
#         PyDetecDiv.app.restoreOverrideCursor()
#
#     def save_rois(self):
#         """
#         Save the areas as ROIs
#         """
#         rois = [item for item in self.scene.items() if isinstance(item, QGraphicsRectItem)]
#         with pydetecdiv_project(PyDetecDiv.project_name) as project:
#             roi_list = [r.name for r in self.image_resource_data.fov.roi_list]
#             for i, rect_item in enumerate(sorted(rois, key=lambda x: x.scenePos().toPoint().toTuple())):
#                 x, y = rect_item.scenePos().toPoint().toTuple()
#                 w, h = rect_item.rect().toRect().getCoords()[2:]
#                 new_roi_name = f'{self.image_resource_data.fov.name}_{x}_{y}_{w}_{h}'
#                 if new_roi_name not in roi_list:
#                     new_roi = ROI(project=project, name=new_roi_name, fov=self.image_resource_data.fov,
#                                   top_left=(x, y), bottom_right=(int(x) + w, int(y) + h))
#                     rect_item.setData(0, new_roi.name)
#         PyDetecDiv().saved_rois.emit(PyDetecDiv.project_name)
#         self.fixate_saved_rois()
#
#     def fixate_saved_rois(self):
#         """
#         Disable the possibility to move ROIs once they have been saved
#         """
#         for r in [item for item in self.scene.items() if isinstance(item, QGraphicsRectItem)]:
#             r.setPen(self.scene.saved_pen)
#             r.setFlag(QGraphicsItem.ItemIsMovable, False)
#
#
# class ViewerScene(QGraphicsScene):
#     """
#     The viewer scene where images and other items are drawn
#     """
#
#     def __init__(self):
#         super().__init__()
#         self.from_x = None
#         self.from_y = None
#         self.pen = QPen(Qt.GlobalColor.cyan, 2)
#         self.match_pen = QPen(Qt.GlobalColor.yellow, 2)
#         self.saved_pen = QPen(Qt.GlobalColor.green, 2)
#         self.warning_pen = QPen(Qt.GlobalColor.red, 2)
#
#     def contextMenuEvent(self, event):
#         """
#         The context menu for area manipulation
#
#         :param event:
#         """
#         menu = QMenu()
#         view_in_new_tab = menu.addAction("View in new tab")
#         r = self.itemAt(event.scenePos(), QTransform().scale(1, 1))
#         if isinstance(r, QGraphicsRectItem):
#             view_in_new_tab.triggered.connect(lambda _: self.parent().view_roi_image(r))
#             PyDetecDiv.app.viewer_roi_click.emit((r, menu))
#             menu.exec(event.screenPos())
#
#     def keyPressEvent(self, event):
#         """
#         Detect when a key is pressed and perform the corresponding action:
#         * QKeySequence.Delete: delete the selected item
#
#         :param event: the key pressed event
#         :type event: QKeyEvent
#         """
#         if event.matches(QKeySequence.Delete):
#             for r in self.selectedItems():
#                 self.removeItem(r)
#
#     def mousePressEvent(self, event):
#         """
#         Detect when the left mouse button is pressed and perform the action corresponding to the currently checked
#         drawing tool
#
#         :param event: the mouse press event
#         :type event: QGraphicsSceneMouseEvent
#         """
#         if event.button() == Qt.LeftButton:
#             match PyDetecDiv.current_drawing_tool:
#                 case DrawingTools.Cursor:
#                     self.select_ROI(event)
#                 case DrawingTools.DrawRect:
#                     self.select_ROI(event)
#                 case DrawingTools.DuplicateItem:
#                     self.duplicate_selected_ROI(event)
#
#     def select_ROI(self, event):
#         """
#         Select the current area/ROI
#
#         :param event: the mouse press event
#         :type event: QGraphicsSceneMouseEvent
#         """
#         _ = [r.setSelected(False) for r in self.items()]
#         r = self.itemAt(event.scenePos(), QTransform().scale(1, 1))
#         if isinstance(r, QGraphicsRectItem):
#             r.setSelected(True)
#             self.display_roi_size(r)
#         if self.selectedItems():
#             self.parent().ui.actionSet_template.setEnabled(True)
#         else:
#             self.parent().ui.actionSet_template.setEnabled(False)
#
#     def get_selected_ROI(self):
#         """
#         Return the selected ROI
#
#         :return: the selected ROI
#         :rtype: QGraphicsRectItem
#         """
#         for selection in self.selectedItems():
#             if isinstance(selection, QGraphicsRectItem):
#                 return selection
#         return None
#
#     def duplicate_selected_ROI(self, event):
#         """
#         Duplicate the currently selected ROI at the current mouse position
#
#         :param event: the mouse press event
#         :type event: QGraphicsSceneMouseEvent
#         """
#         pos = event.scenePos()
#         roi = self.get_selected_ROI()
#         if roi:
#             roi = self.addRect(roi.rect())
#             roi.setPen(self.pen)
#             w, h = roi.rect().size().toTuple()
#             roi.setPos(QPoint(pos.x() - np.around(w / 2.0), pos.y() - np.around(h / 2.0)))
#             roi.setFlags(QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemIsSelectable)
#             roi.setData(0, f'Region{len(self.items())}')
#             self.select_ROI(event)
#             if [r for r in roi.collidingItems(Qt.IntersectsItemBoundingRect) if isinstance(r, QGraphicsRectItem)]:
#                 roi.setPen(self.warning_pen)
#             else:
#                 roi.setPen(self.pen)
#
#     def mouseMoveEvent(self, event):
#         """
#         Detect mouse movement and apply the appropriate method according to the currently checked drawing tool and key
#         modifier
#
#         :param event: the mouse move event
#         :type event: QGraphicsSceneMouseEvent
#         """
#         if event.button() == Qt.NoButton:
#             match PyDetecDiv.current_drawing_tool, event.modifiers():
#                 case DrawingTools.Cursor, Qt.NoModifier:
#                     self.move_ROI(event)
#                 case DrawingTools.Cursor, Qt.ControlModifier:
#                     self.draw_ROI(event)
#                 case DrawingTools.DrawRect, Qt.NoModifier:
#                     self.draw_ROI(event)
#                 case DrawingTools.DrawRect, Qt.ControlModifier:
#                     self.move_ROI(event)
#                 case DrawingTools.DuplicateItem, Qt.NoModifier:
#                     self.move_ROI(event)
#
#     def move_ROI(self, event):
#         """
#         Move the currently selected ROI if it is movable
#
#         :param event: the mouse move event
#         :type event: QGraphicsSceneMouseEvent
#         """
#         roi = self.get_selected_ROI()
#         if roi and (roi.flags() & QGraphicsItem.ItemIsMovable):
#             pos = event.scenePos()
#             roi.moveBy(pos.x() - event.lastScenePos().x(), pos.y() - event.lastScenePos().y())
#             if [r for r in roi.collidingItems(Qt.IntersectsItemBoundingRect) if isinstance(r, QGraphicsRectItem)]:
#                 roi.setPen(self.warning_pen)
#             else:
#                 roi.setPen(self.pen)
#
#     def draw_ROI(self, event):
#         """
#         Draw or redraw the currently selected ROI if it is movable
#
#         :param event: the mouse press event
#         :type event: QGraphicsSceneMouseEvent
#         """
#         roi = self.get_selected_ROI()
#         pos = event.scenePos()
#         if roi and (roi.flags() & QGraphicsItem.ItemIsMovable):
#             roi_pos = roi.scenePos()
#             w, h = np.max([round_to_even(pos.x() - roi_pos.x()), 5]), np.max([round_to_even(pos.y() - roi_pos.y()), 5])
#             rect = QRect(0, 0, w, h)
#             roi.setRect(rect)
#         else:
#             roi = self.addRect(QRect(0, 0, 5, 5))
#             roi.setPen(self.pen)
#             roi.setPos(QPoint(pos.x(), pos.y()))
#             roi.setFlags(QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemIsSelectable)
#             roi.setData(0, f'Region{len(self.items())}')
#             self.select_ROI(event)
#         if [r for r in roi.collidingItems(Qt.IntersectsItemBoundingRect) if isinstance(r, QGraphicsRectItem)]:
#             roi.setPen(self.warning_pen)
#         else:
#             roi.setPen(self.pen)
#         self.display_roi_size(roi)
#
#     def set_ROI_width(self, width):
#         roi = self.get_selected_ROI()
#         if roi and (roi.flags() & QGraphicsItem.ItemIsMovable):
#             rect = QRect(0, 0, width, roi.rect().height())
#             roi.setRect(rect)
#
#     def set_Item_width(self, width):
#         self.set_ROI_width(width)
#
#     def set_ROI_height(self, height):
#         roi = self.get_selected_ROI()
#         if roi and (roi.flags() & QGraphicsItem.ItemIsMovable):
#             rect = QRect(0, 0, roi.rect().width(), height)
#             roi.setRect(rect)
#
#     def set_Item_height(self, height):
#         self.set_ROI_height(height)
#
#     def display_roi_size(self, roi):
#         PyDetecDiv.main_window.drawing_tools.roi_width.setValue(roi.rect().width())
#         PyDetecDiv.main_window.drawing_tools.roi_height.setValue(roi.rect().height())
#
#     def display_Item_size(self, item):
#         self.display_roi_size(item)
