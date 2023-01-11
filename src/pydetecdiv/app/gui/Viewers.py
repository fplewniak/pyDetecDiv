#  CeCILL FREE SOFTWARE LICENSE AGREEMENT Version 2.1 dated 2013-06-21
#  Frédéric PLEWNIAK, CNRS/Université de Strasbourg UMR7156 - GMGM
import dearpygui.dearpygui as dpg

class ImageViewer():
    def __init__(self):
        self.active = False

    def clear(self):
        if self.active:
            dpg.delete_item('image_registry')
            dpg.delete_item('image_viewer')
            dpg.delete_item('image_control')
            self.active = False

    def imshow(self, image_resource, c=0, z=0, t=0):
        width, height, channels, data = image_resource.as_texture(c=c, z=z, t=t)
        with dpg.texture_registry(show=False, tag='image_registry') as registry:
            dpg.add_dynamic_texture(width=width, height=height, default_value=data, tag="texture_tag")
        with dpg.group(parent='viewer_window', tag='image_control'):
            with dpg.group(horizontal=True):
                dpg.add_text('T')
                dpg.add_slider_int(max_value=image_resource.shape[1] - 1, clamped=True, width=-1,
                               callback=change_time, user_data={'image_resource': image_resource, 'c': c, 'z': z})
            with dpg.group(horizontal=True):
                dpg.add_text('Z')
                dpg.add_slider_int(max_value=image_resource.shape[2] - 1, clamped=True, width=-1,
                               callback=change_layer, user_data={'image_resource': image_resource, 'c': c, 't': t})
            with dpg.group(horizontal=True):
                dpg.add_text('C')
                dpg.add_slider_int(max_value=image_resource.shape[0] - 1, clamped=True, width=-1,
                               callback=change_channel, user_data={'image_resource': image_resource, 't': t, 'z': z})
        with dpg.plot(label="Image viewer", tag='image_viewer', height=-1, equal_aspects=True, width=-1,
                      parent='viewer_window'):
            dpg.add_plot_axis(dpg.mvXAxis, )
            with dpg.plot_axis(dpg.mvYAxis, ):
                dpg.add_image_series("texture_tag", [0, 0], [width, height], label="image", tag='image')
        self.active = True

def change_time(sender, app_data, user_data):
    width, height, channels, data = user_data['image_resource'].as_texture(c=user_data['c'], t=app_data, z=user_data['z'],)
    dpg.set_value("texture_tag", data)

def change_layer(sender, app_data, user_data):
    width, height, channels, data = user_data['image_resource'].as_texture(c=user_data['c'], t=user_data['t'], z=app_data,)
    dpg.set_value("texture_tag", data)

def change_channel(sender, app_data, user_data):
    width, height, channels, data = user_data['image_resource'].as_texture(c=app_data, t=user_data['t'], z=user_data['z'],)
    dpg.set_value("texture_tag", data)
