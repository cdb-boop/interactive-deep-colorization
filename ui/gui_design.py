from PyQt5.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QGroupBox, QPushButton, QCheckBox
from PyQt5.QtGui import QKeyEvent
from PyQt5.QtCore import Qt
import numpy as np
from . import gui_draw
from . import gui_visualize
from . import gui_gamut
from . import gui_palette
from . import utils
from data import colorize_image
import time


class GUIDesign(QWidget):
    def __init__(
            self, 
            color_model: colorize_image.ColorizeImageTorch | colorize_image.ColorizeImageCaffe, 
            dist_model: colorize_image.ColorizeImageTorchDist | colorize_image.ColorizeImageCaffeDist | None = None, 
            img_file: str | None = None, 
            load_size: int = 256, 
            win_size: int = 256):
        QWidget.__init__(self)

        # main layout
        main_layout = QHBoxLayout()
        self.setLayout(main_layout)

        # gamut layout
        self.gamut = gui_gamut.GUIGamut(gamut_size=160)
        gamutLayout = self.AddWidget(self.gamut, 'ab Color Gamut')
        color_layout = QVBoxLayout()

        color_layout.addLayout(gamutLayout)
        main_layout.addLayout(color_layout)

        # palettes
        self.suggestion_palette = gui_palette.GUIPalette(grid_sz=(10, 1))
        suggestion_palette_layout = self.AddWidget(self.suggestion_palette, 'Suggested colors')
        color_layout.addLayout(suggestion_palette_layout)

        self.recently_used_palette = gui_palette.GUIPalette(grid_sz=(10, 1))
        recently_used_palette_layout = self.AddWidget(self.recently_used_palette, 'Recently used colors')
        color_layout.addLayout(recently_used_palette_layout)

        # color indicator
        self.color_indicator = QPushButton()  # to visualize the selected color
        self.color_indicator.setFixedWidth(self.suggestion_palette.width())
        self.color_indicator.setFixedHeight(25)
        self.color_indicator_reset()
        color_indicator_layout = self.AddWidget(self.color_indicator, 'Color')
        color_layout.addLayout(color_indicator_layout)
        color_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # drawPad layout
        drawpad_layout = QVBoxLayout()
        main_layout.addLayout(drawpad_layout)
        self.drawpad = gui_draw.GUIDraw(color_model, dist_model, load_size=load_size, win_size=win_size)
        drawpad_layout = self.AddWidget(self.drawpad, 'Drawing Pad')
        main_layout.addLayout(drawpad_layout)

        drawpad_menu = QHBoxLayout()

        self.toggle_gray_drawpad = QCheckBox("&Gray")
        self.toggle_gray_drawpad.setToolTip('show gray-scale image')

        self.load_button = QPushButton('&Load')
        self.load_button.setToolTip('load an input image')
        self.save_button = QPushButton("&Save")
        self.save_button.setToolTip('Save the current result.')

        drawpad_menu.addWidget(self.toggle_gray_drawpad)
        drawpad_menu.addWidget(self.load_button)
        drawpad_menu.addWidget(self.save_button)

        drawpad_layout.addLayout(drawpad_menu)
        self.visualization = gui_visualize.GUIVisualize(win_size=win_size, scale=win_size / float(load_size))
        visualization_layout = self.AddWidget(self.visualization, 'Result')
        main_layout.addLayout(visualization_layout)

        self.restart_button = QPushButton("&Restart")
        self.restart_button.setToolTip('Restart the system')

        self.quit_button = QPushButton("&Quit")
        self.quit_button.setToolTip('Quit the system.')
        visualization_menu = QHBoxLayout()
        visualization_menu.addWidget(self.restart_button)

        visualization_menu.addWidget(self.quit_button)
        visualization_layout.addLayout(visualization_menu)

        self.drawpad.update()
        self.visualization.update()

        # drawpad events
        self.visualization.color_selected.connect(self.drawpad.set_color)
        self.gamut.color_selected.connect(self.drawpad.set_color)
        self.suggestion_palette.color_selected.connect(self.drawpad.set_color)
        self.recently_used_palette.color_selected.connect(self.drawpad.set_color)

        # colorized image visualization events
        self.drawpad.colorized_image_generated.connect(self.visualization.set_image)

        # gamut events
        self.drawpad.gamut_changed.connect(self.gamut.set_gamut)

        self.drawpad.selected_color_changed.connect(self.gamut.set_ab)
        self.visualization.color_selected.connect(self.gamut.set_ab)
        self.suggestion_palette.color_selected.connect(self.gamut.set_ab)
        self.recently_used_palette.color_selected.connect(self.gamut.set_ab)

        # suggestion palette events
        self.drawpad.suggested_colors_changed.connect(self.suggestion_palette.set_colors)

        # recently used palette events
        self.drawpad.recently_used_colors_changed.connect(self.recently_used_palette.set_colors)

        # color indicator events
        self.drawpad.selected_color_changed.connect(self.set_indicator_color)
        self.visualization.color_selected.connect(self.set_indicator_color)
        self.gamut.color_selected.connect(self.set_indicator_color)
        self.suggestion_palette.color_selected.connect(self.set_indicator_color)
        self.recently_used_palette.color_selected.connect(self.set_indicator_color)

        # menu events
        self.toggle_gray_drawpad.setChecked(True)
        self.restart_button.clicked.connect(self.reset)
        self.quit_button.clicked.connect(self.quit)
        self.toggle_gray_drawpad.toggled.connect(self.enable_gray)
        self.save_button.clicked.connect(self.save)
        self.load_button.clicked.connect(self.load)

        self.start_t = time.time()

        if img_file is not None:
            self.drawpad.init_result(img_file)

    def AddWidget(self, widget: QWidget, title: str) -> QVBoxLayout:
        layout = QVBoxLayout()
        box = QGroupBox()
        box.setTitle(title)
        vertical_box_t = QVBoxLayout()
        vertical_box_t.addWidget(widget)
        box.setLayout(vertical_box_t)
        layout.addWidget(box)

        return layout

    def nextImage(self) -> None:
        self.drawpad.nextImage()

    def reset(self) -> None:
        # self.start_t = time.time()
        print('============================reset all=========================================')
        self.visualization.reset()
        self.gamut.reset()
        self.suggestion_palette.reset()
        self.recently_used_palette.reset()
        self.drawpad.reset()
        self.color_indicator_reset()
        self.update()

    def enable_gray(self) -> None:
        self.drawpad.enable_gray()

    def quit(self) -> None:
        print(f"GUIDesign: time spent = {time.time() - self.start_t:3.3f}")
        self.close()

    def save(self) -> None:
        print(f"GUIDesign: time spent = {time.time() - self.start_t:3.3f}")
        self.drawpad.save_result()

    def load(self) -> None:
        self.drawpad.load_image()

    def color_indicator_reset(self) -> None:
        self.set_indicator_color(np.array((0,0,0)).astype('uint8'))

    def set_indicator_color(self, color: np.ndarray) -> None:
        color = utils.ndarray_to_qcolor(color)
        self.color_indicator.setStyleSheet(f"background-color: {color.name()}")

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_R:
            self.reset()

        if event.key() == Qt.Key.Key_Q:
            self.save()
            self.quit()

        if event.key() == Qt.Key.Key_S:
            self.save()

        if event.key() == Qt.Key.Key_G:
            self.toggle_gray_drawpad.toggle()

        if event.key() == Qt.Key.Key_L:
            self.load()
