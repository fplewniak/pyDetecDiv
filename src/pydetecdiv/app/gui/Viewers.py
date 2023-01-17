#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
"""
Classes to handle data viewers for visualization of images, tables, plots, etc
"""
import dearpygui.dearpygui as dpg


class ImageViewer:
    """
    Image viewer to visualize image data
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.active = False
        self.image_resource = None

    def clear(self):
        """
        Clear the image viewer if it was in use
        :return: the Image viewer object
        :rtype: ImageViewer
        """
        if self.active:
            self.active = False
            dpg.delete_item('image_registry')
            dpg.delete_item('image_viewer')
            dpg.delete_item('image_control')
            self.image_resource = None
        return self

    def imshow(self, image_resource, c=0, z=0, t=0):
        """
        Show an image contained in image resource for the specified channel, layer and frame
        :param image_resource: the image resource containing the image data
        :type image_resource: MemMapTiff or MultipleTiff object
        :param c: the channel index
        :type c: int
        :param z: the layer index
        :type z: int
        :param t: the frame index
        :type t: int
        :return: the ImageViewer object
        :rtype: ImageViewer
        """
        self.image_resource = image_resource
        width, height, _, data = image_resource.as_texture(c=c, z=z, t=t)
        with dpg.texture_registry(show=False, tag='image_registry'):
            dpg.add_dynamic_texture(width=width, height=height, default_value=data, tag="image_to_view")
        with dpg.group(parent='viewer_window', tag='image_control'):
            with dpg.group(horizontal=True):
                dpg.add_text('T')
                dpg.add_slider_int(tag='t_slider', max_value=image_resource.sizeT - 1, clamped=True, width=-1,
                                   callback=self.update_image)
            with dpg.group(horizontal=True):
                dpg.add_text('Z')
                dpg.add_slider_int(tag='z_slider', max_value=image_resource.sizeZ - 1, clamped=True, width=-1,
                                   callback=self.update_image)
            with dpg.group(horizontal=True):
                dpg.add_text('C')
                dpg.add_slider_int(tag='c_slider', max_value=image_resource.sizeC - 1, clamped=True, width=-1,
                                   callback=self.update_image)
        with dpg.plot(tag='image_viewer', height=-1, equal_aspects=True, width=-1,
                      parent='viewer_window'):
            dpg.add_plot_axis(dpg.mvXAxis, )
            with dpg.plot_axis(dpg.mvYAxis, ):
                dpg.add_image_series("image_to_view", [0, 0], [width, height], label="image", tag='image')
        self.active = True
        return self

    def update_image(self, _, app_data, user_data):
        """
        Callback method to change display to another frame
        :param _: dummy sender parameter that is not used
        :param app_data: the frame index
        :type app_data: int
        :param user_data: channel and layer indices
        :type user_data: dict
        """
        _, _, _, data = self.image_resource.as_texture(c=dpg.get_value('c_slider'),
                                                       t=dpg.get_value('t_slider'), z=dpg.get_value('z_slider'))
        dpg.set_value("image_to_view", data)

    def run_video(self):
        for t in range(0, self.image_resource.sizeT):
            dpg.set_value('t_slider', t)
            self.update_image(None, None, None)
