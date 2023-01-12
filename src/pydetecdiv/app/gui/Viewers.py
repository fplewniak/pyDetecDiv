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
            dpg.delete_item('image_registry')
            dpg.delete_item('image_viewer')
            dpg.delete_item('image_control')
            self.active = False
            self.image_resource = None
        return self

    def imshow(self, image_resource, c=0, z=0, t=0):
        """
        Show an image contained in image resource for the specified channel, layer and frame
        :param image_resource: the image resource containing the image data
        :type image_resource: SingleTiff or MultipleTiff object
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
                dpg.add_slider_int(max_value=image_resource.shape[1] - 1, clamped=True, width=-1,
                                   callback=self.change_time, user_data={'c': c, 'z': z})
            with dpg.group(horizontal=True):
                dpg.add_text('Z')
                dpg.add_slider_int(max_value=image_resource.shape[2] - 1, clamped=True, width=-1,
                                   callback=self.change_layer, user_data={'c': c, 't': t})
            with dpg.group(horizontal=True):
                dpg.add_text('C')
                dpg.add_slider_int(max_value=image_resource.shape[0] - 1, clamped=True, width=-1,
                                   callback=self.change_channel, user_data={'t': t, 'z': z})
        with dpg.plot(tag='image_viewer', height=-1, equal_aspects=True, width=-1,
                      parent='viewer_window'):
            dpg.add_plot_axis(dpg.mvXAxis, )
            with dpg.plot_axis(dpg.mvYAxis, ):
                dpg.add_image_series("image_to_view", [0, 0], [width, height], label="image", tag='image')
        self.active = True
        return self

    def change_time(self, _, app_data, user_data):
        """
        Callback method to change display to another frame
        :param _: dummy sender parameter that is not used
        :param app_data: the frame index
        :type app_data: int
        :param user_data: channel and layer indices
        :type user_data: dict
        """
        _, _, _, data = self.image_resource.as_texture(c=user_data['c'], t=app_data, z=user_data['z'], )
        dpg.set_value("image_to_view", data)

    def change_layer(self, _, app_data, user_data):
        """
        Callback method to change display to another layer
        :param _: dummy sender parameter that is not used
        :param app_data: the layer index
        :type app_data: int
        :param user_data: channel and frame indices
        :type user_data: dict
        """
        _, _, _, data = self.image_resource.as_texture(c=user_data['c'], t=user_data['t'], z=app_data, )
        dpg.set_value("image_to_view", data)

    def change_channel(self, _, app_data, user_data):
        """
        Callback method to change display to another channel
        :param _: dummy sender parameter that is not used
        :param app_data: the channel index
        :type app_data: int
        :param user_data: frame and layer indices
        :type user_data: dict
        """
        _, _, _, data = self.image_resource.as_texture(c=app_data, t=user_data['t'], z=user_data['z'], )
        dpg.set_value("image_to_view", data)
