from PySide6.QtGui import QColor


class Colours:
    """
    Colour definition for brushes used to display Mask graphics items
    """
    palette = [
        QColor('black'),
        QColor('red'),
        QColor('green'),
        QColor('blue'),
        QColor('yellow'),
        QColor('magenta'),
        QColor('cyan'),
        QColor('darkgray'),
        QColor('darkred'),
        QColor('darkgreen'),
        QColor('darkblue'),
        QColor('darkyellow'),
        QColor('darkmagenta'),
        QColor('darkcyan'),
        QColor('lightgray'),
        ]
    palette_off = [
        QColor(0, 0, 0, 255),
        QColor(255, 0, 0, 255),
        QColor(0, 255, 0, 255),
        QColor(0, 0, 255, 255),
        QColor(255, 255, 0, 255),
        QColor(255, 0, 255, 255),
        QColor(0, 255, 255, 255),
        QColor(255, 128, 0, 255),
        QColor(255, 0, 128, 255),
        QColor(128, 255, 0, 255),
        QColor(128, 0, 255, 255),
        QColor(0, 255, 128, 255),
        QColor(0, 128, 255, 255),
        ]
