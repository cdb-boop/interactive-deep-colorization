import numpy as np
import numpy.typing as npt
import cv2
from PyQt5.QtWidgets import QWidget, QApplication, QFileDialog
from PyQt5.QtGui import QColor, QPainter, QPaintEvent, QImage, QWheelEvent, QMouseEvent
from PyQt5.QtCore import Qt, pyqtSignal, QSize, QPoint
from .ui_control import UIControl
from . import utils

from data import lab_gamut, colorize_image as CI
import importlib
skcolor = importlib.import_module("skimage.color")  # ignore import issue with skimage
import os
import datetime
import glob
import sys
import warnings
from enum import Enum

class UIModes(Enum):
    NONE = 0
    ERASE = 1
    POINT = 2
    STROKE = 3

class GUIDraw(QWidget):
    gamut_changed = pyqtSignal(np.float64)
    suggested_colors_changed = pyqtSignal(np.ndarray)  # Nx3 float32
    recently_used_colors_changed = pyqtSignal(np.ndarray)  # Nx3 float32
    selected_color_changed = pyqtSignal(np.ndarray)  # 1x3 uint8
    colorized_image_generated = pyqtSignal(np.ndarray)  # NxNx3 uint8

    def __init__(
            self, 
            colorizer: CI.Colorizer,
            dist_colorizer: CI.ColorizerDist | None = None,
            load_size: int = 256,
            win_size: int = 512):
        QWidget.__init__(self)
        self.image_file: None | str = None
        self.pos: None | QPoint = None
        self.colorizer = colorizer
        self.dist_colorizer = dist_colorizer
        self.win_size = win_size
        self.load_size = load_size
        self.setFixedSize(win_size, win_size)
        self.uiControl = UIControl(win_size=win_size, load_size=load_size)
        self.move(win_size, win_size)
        self.movie = True
        self.init_color()
        self.im_gray3: None | npt.NDArray[np.uint8] = None
        self.eraseMode = False
        self.ui_mode = UIModes.NONE
        self.image_loaded = False  # TODO: is this redundant?
        self.use_gray = True
        self.total_images = 0
        self.image_id = 0

    def clock_count(self) -> None:
        self.count_secs -= 1
        self.update()

    def init_result(self, image_file: str) -> None:
        self.read_image(image_file)  # read an image
        self.reset()

    def get_batches(self, img_dir: str) -> None:
        self.img_list = glob.glob(os.path.join(img_dir, '*.JPEG'))
        self.total_images = len(self.img_list)
        img_first = self.img_list[0]
        self.init_result(img_first)

    def nextImage(self) -> None:
        self.save_result()
        self.image_id += 1
        if self.image_id == self.total_images:
            print("GUIDraw: You have finished all the results")
            sys.exit()
        img_current = self.img_list[self.image_id]
        # self.reset()
        self.init_result(img_current)
        self.reset_timer()

    def read_image(self, image_file: str) -> None:
        # open image
        print(f"GUIDraw: Open image {image_file}")
        im_bgr = cv2.imread(image_file)
        self.im_full = im_bgr.copy()
        # self.result = None
        self.image_loaded = True
        self.image_file = image_file
        # get image for display
        shape = self.im_full.shape
        h, w = int(shape[0]), int(shape[1])
        max_width = max(h, w)
        r = self.win_size / float(max_width)
        self.scale = float(self.win_size) / self.load_size
        print(f"GUIDraw: Image scale {self.scale}")
        rw = int(round(r * w / 4.0) * 4)
        rh = int(round(r * h / 4.0) * 4)
        print("\n")

        self.im_win = cv2.resize(self.im_full, (rw, rh), interpolation=cv2.INTER_CUBIC)

        self.dw = int((self.win_size - rw) // 2)
        self.dh = int((self.win_size - rh) // 2)
        self.win_w = rw
        self.win_h = rh
        self.uiControl.setImageSize((rw, rh))
        im_gray = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2GRAY)
        self.im_gray3 = cv2.cvtColor(im_gray, cv2.COLOR_GRAY2BGR)

        self.gray_win = cv2.resize(self.im_gray3, (rw, rh), interpolation=cv2.INTER_CUBIC)
        im_bgr = cv2.resize(im_bgr, (self.load_size, self.load_size), interpolation=cv2.INTER_CUBIC)
        self.im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
        lab_win = skcolor.rgb2lab(self.im_win[:, :, ::-1])

        self.im_lab = skcolor.rgb2lab(im_bgr[:, :, ::-1])
        self.im_l = self.im_lab[:, :, 0]
        self.l_win = lab_win[:, :, 0]
        self.im_ab = self.im_lab[:, :, 1:]
        self.im_size = self.im_rgb.shape[0:2]

        self.im_ab0 = np.zeros((2, self.load_size, self.load_size))
        self.im_mask0 = np.zeros((1, self.load_size, self.load_size))
        self.brushWidth = 2 * self.scale

        self.colorizer.data.load_image(image_file)

        if (self.dist_colorizer is not None):
            self.dist_colorizer.data.set_image(self.im_rgb)
            self.predict_color()

    def update_im(self) -> None:
        self.update()
        QApplication.processEvents()

    def update_ui(self, move_point: bool = True) -> bool:
        if self.ui_mode == UIModes.NONE:
            return False

        is_predict = False
        snap_qcolor = self.calibrate_color(self.user_color, self.pos)
        self.color = snap_qcolor
        self.selected_color_changed.emit(utils.qcolor_to_ndarray(self.color))
        if self.ui_mode == UIModes.POINT:
            if move_point:
                self.uiControl.movePoint(self.pos, snap_qcolor, self.user_color, self.brushWidth)
            else:
                self.user_color, self.brushWidth, isNew = self.uiControl.addPoint(self.pos, snap_qcolor, self.user_color, self.brushWidth)
                if isNew:
                    is_predict = True
                    # self.predict_color()
        elif self.ui_mode == UIModes.STROKE:
            self.uiControl.addStroke(self.prev_pos, self.pos, snap_qcolor, self.user_color, self.brushWidth)
        elif self.ui_mode == UIModes.ERASE:
            isRemoved = self.uiControl.erasePoint(self.pos)
            if isRemoved:
                is_predict = True
                # self.predict_color()
        return is_predict

    def reset(self) -> None:
        self.ui_mode = UIModes.NONE
        self.pos = None
        self.result = None
        self.user_color = None
        self.color = None
        self.uiControl.reset()
        self.init_color()
        self.compute_colorized_image()
        self.predict_color()
        self.update()

    def scale_point(self, pnt: QPoint) -> tuple[int, int]:
        x = int((pnt.x() - self.dw) / float(self.win_w) * self.load_size)
        y = int((pnt.y() - self.dh) / float(self.win_h) * self.load_size)
        return x, y

    def valid_point(self, pnt: QPoint) -> QPoint | None:
        if pnt is None:
            warnings.warn(f"GUIDraw: 'pnt' was 'None'", RuntimeWarning)
            return None

        if pnt.x() >= self.dw and pnt.y() >= self.dh and pnt.x() < self.win_size - self.dw and pnt.y() < self.win_size - self.dh:
            x = int(np.round(pnt.x()))
            y = int(np.round(pnt.y()))
            return QPoint(x, y)
        else:
            return None

    def init_color(self) -> None:
        self.user_color = QColor(128, 128, 128)  # default color red
        self.color = self.user_color

    def signal_ui_changes(self, pos: QPoint) -> None:
        if not isinstance(pos, QPoint):
            warnings.warn("GUIDraw: 'pos' was not 'QPoint'", RuntimeWarning)
            return

        x, y = self.scale_point(pos)
        L = self.im_lab[y, x, 0]
        self.gamut_changed.emit(L)

        rgb_colors = self.suggest_color(h=y, w=x, K=9)
        if rgb_colors.shape[0] > 0:
            rgb_colors[-1, :] = 0.5
        self.suggested_colors_changed.emit(rgb_colors)

        recently_used_colors = self.uiControl.get_recently_used_colors()
        self.recently_used_colors_changed.emit(recently_used_colors)

        snap_color = self.calibrate_color(self.user_color, pos)
        c = utils.qcolor_to_ndarray(snap_color)
        self.selected_color_changed.emit(c)

    def calibrate_color(self, c: QColor, pos: QPoint) -> QColor:
        x, y = self.scale_point(pos)

        # snap color based on L color
        color_array = utils.qcolor_to_ndarray(c)
        mean_L = self.im_l[y, x]
        snap_color = lab_gamut.snap_ab(mean_L, color_array)
        snap_qcolor = utils.ndarray_to_qcolor(snap_color)
        return snap_qcolor

    def set_color(self, c_rgb: np.ndarray) -> None:
        if self.pos is None:
            warnings.warn("GUIDraw: 'c_rgb' is 'None'", RuntimeWarning)
            return

        print("GUIDraw: Set color")

        c = utils.ndarray_to_qcolor(c_rgb)
        self.user_color = c
        
        snap_qcolor = self.calibrate_color(c, self.pos)
        self.color = snap_qcolor

        self.uiControl.update_color(snap_qcolor, self.user_color)
        self.compute_colorized_image()

    def erase(self) -> None:
        self.eraseMode = not self.eraseMode

    def load_image(self) -> None:
        img_path = QFileDialog.getOpenFileName(self, 'load an input image')[0]
        if os.path.isfile(img_path):
            self.init_result(img_path)

    def save_result(self) -> None:
        if self.image_file is None:
            warnings.warn("GUIDraw: Attempted to save 'None' image", RuntimeWarning)
            return
        path = os.path.abspath(self.image_file)
        path, ext = os.path.splitext(path)

        suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
        method_str = 'with_dist' if self.dist_colorizer is not None else 'without_dist'
        save_path = "_".join([path, method_str, suffix])

        print(f"GUIDraw: Saving result to \"{save_path}\"\n")
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        np.save(os.path.join(save_path, 'im_l.npy'), self.colorizer.data.img_l)
        np.save(os.path.join(save_path, 'im_ab.npy'), self.im_ab0)
        np.save(os.path.join(save_path, 'im_mask.npy'), self.im_mask0)

        result_bgr = cv2.cvtColor(self.result, cv2.COLOR_RGB2BGR)
        mask = self.im_mask0.transpose((1, 2, 0)).astype(np.uint8) * 255
        cv2.imwrite(os.path.join(save_path, 'input_mask.png'), mask)
        cv2.imwrite(os.path.join(save_path, 'ours.png'), result_bgr)
        cv2.imwrite(os.path.join(save_path, 'ours_fullres.png'), self.colorizer.data.get_img_fullres()[:, :, ::-1])
        cv2.imwrite(os.path.join(save_path, 'input_fullres.png'), self.colorizer.data.get_input_img_fullres()[:, :, ::-1])
        cv2.imwrite(os.path.join(save_path, 'input.png'), self.colorizer.data.get_input_img()[:, :, ::-1])
        cv2.imwrite(os.path.join(save_path, 'input_ab.png'), self.colorizer.data.get_sup_img()[:, :, ::-1])

    def enable_gray(self) -> None:
        self.use_gray = not self.use_gray
        self.update()

    def predict_color(self) -> None:
        if self.dist_colorizer is not None and self.image_loaded:
            im, mask = self.uiControl.get_input()
            im_mask0 = mask > 0.0
            self.im_mask0 = im_mask0.transpose((2, 0, 1))
            im_lab = skcolor.rgb2lab(im).transpose((2, 0, 1))
            self.im_ab0 = im_lab[1:3, :, :]

            result = self.dist_colorizer.run(self.im_ab0, self.im_mask0)
            if isinstance(result, Exception):
                warnings.warn(f"GUIDraw: {result}", RuntimeWarning)

    def suggest_color(self, h: int, w: int, K: int = 5) -> npt.NDArray[np.float32]:
        if self.dist_colorizer is not None and self.image_loaded:
            result = self.dist_colorizer.get_ab_reccs(h, w, K, 25000)
            if isinstance(result, Exception):
                warnings.warn(f"GUIDraw: {result}", RuntimeWarning)
                return np.ndarray([], np.float32)
            ab, conf = result
            if ab is None:
                warnings.warn("GUIDraw: Unable to suggest color", RuntimeWarning)
                return np.ndarray([], np.float32)
            L = np.tile(self.im_lab[h, w, 0], (K, 1))
            colors_lab = np.concatenate((L, ab), axis=1)
            colors_lab3 = colors_lab[:, np.newaxis, :]
            colors_rgb = np.clip(np.squeeze(skcolor.lab2rgb(colors_lab3)), 0, 1)
            colors_rgb_withcurr = np.concatenate((self.colorizer.data.get_img_forward()[h, w, np.newaxis, :] / 255., colors_rgb), axis=0)
            return colors_rgb_withcurr
        else:
            warnings.warn("GUIDraw: No color suggestion returned", RuntimeWarning)
            return np.ndarray([], np.float32)

    def compute_colorized_image(self) -> None:
        im, mask = self.uiControl.get_input()
        im_mask0 = mask > 0.0
        self.im_mask0 = im_mask0.transpose((2, 0, 1))
        im_lab = skcolor.rgb2lab(im).transpose((2, 0, 1))
        self.im_ab0 = im_lab[1:3, :, :]

        result = self.colorizer.run(self.im_ab0, self.im_mask0)
        if isinstance(result, Exception):
            warnings.warn(f"GUIDraw: {result}", RuntimeWarning)
            return
        ab = self.colorizer.data.output_ab.transpose((1, 2, 0))
        ab_win = cv2.resize(ab, (self.win_w, self.win_h), interpolation=cv2.INTER_CUBIC)
        pred_lab = np.concatenate((self.l_win[..., np.newaxis], ab_win), axis=2)
        pred_rgb = (np.clip(skcolor.lab2rgb(pred_lab), 0, 1) * 255).astype('uint8')
        self.result = pred_rgb
        self.colorized_image_generated.emit(self.result)
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter()
        painter.begin(self)
        painter.fillRect(event.rect(), QColor(49, 54, 49))
        painter.setRenderHint(QPainter.Antialiasing)
        if self.use_gray or self.result is None:
            im = self.gray_win
        else:
            im = self.result

        if im is not None:
            qImg = QImage(im.tostring(), im.shape[1], im.shape[0], QImage.Format_RGB888)
            painter.drawImage(self.dw, self.dh, qImg)

        self.uiControl.update_painter(painter)
        painter.end()

    def wheelEvent(self, event: QWheelEvent) -> None:
        d = event.angleDelta().y() / 120
        self.brushWidth = min(4.05 * self.scale, max(0, self.brushWidth + d * self.scale))
        print(f"GUIDraw: Brush width {self.brushWidth}")
        _ = self.update_ui(move_point=True)
        self.update()

    #def is_same_point(self, pos1: QPoint, pos2: QPoint) -> bool:
    #    if pos1 is None or pos2 is None:
    #        warnings.warn(f"GUIDraw: Attempted to compare point with 'None' type", RuntimeWarning)
    #        return False
    #    dx = pos1.x() - pos2.x()
    #    dy = pos1.y() - pos2.y()
    #    d = dx * dx + dy * dy
    #    # print(f"GUIDraw: Distance between points = {d}")
    #    return d < 25

    def mousePressEvent(self, event: QMouseEvent) -> None:
        print(f"GUIDraw: Mouse press ({event.pos().x()}, {event.pos().y()})")
        pos = self.valid_point(event.pos())
        if pos is None:
            return

        if event.button() == Qt.MouseButton.LeftButton:
            self.pos = pos
            self.ui_mode = UIModes.POINT
            _ = self.update_ui(move_point=False)
            self.signal_ui_changes(pos)
            self.compute_colorized_image()
        elif event.button() == Qt.MouseButton.RightButton:
            # draw the stroke
            self.pos = pos
            self.ui_mode = UIModes.ERASE
            _ = self.update_ui(move_point=False)
            self.compute_colorized_image()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        self.pos = self.valid_point(event.pos())
        if self.pos is not None:
            if self.ui_mode == UIModes.POINT:
                _ = self.update_ui(move_point=True)
                self.compute_colorized_image()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        pass

    def sizeHint(self) -> QSize:
        return QSize(self.win_size, self.win_size)  # 28 * 8
