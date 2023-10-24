from pydetecdiv import plugins
from pydetecdiv.plugins.example.Actions import Action1, Action2

class Plugin(plugins.Plugin):
    id = 'gmgm.plewniak.example'
    version = '1.0.0'
    name = 'Example'
    category = 'Plugin examples'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print(f'Creating {self.name} plugin')

    # def register_object(self, obj):
    #     match obj.__class__.__name__:
    #         case 'MainWindow':
    #             print('Modify main window gui')
    #         case _:
    #             print(f'Do nothing with object of class {obj.__class__.__name__}')

    def addActions(self, menu):
        Action1(menu)
        Action2(menu)


