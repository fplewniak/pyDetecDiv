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
            self.active = False

    def imshow(self, image_resource, c=0, z=0, t=0):
        width, height, channels, data = image_resource.as_texture()
        with dpg.texture_registry(show=False, tag='image_registry') as registry:
            dpg.add_dynamic_texture(width=width, height=height, default_value=data, tag="texture_tag")
        with dpg.plot(label="Image viewer", tag='image_viewer', height=-1, equal_aspects=True, width=-1, parent='viewer_window'):
            dpg.add_plot_axis(dpg.mvXAxis, )
            with dpg.plot_axis(dpg.mvYAxis, ):
                dpg.add_image_series("texture_tag", [0, 0], [1000, 1000], label="image", tag='image')
        self.active = True
