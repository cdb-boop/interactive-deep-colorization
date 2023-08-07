import numpy as np
import cv2
from PyQt5.QtWidgets import QWidget, QApplication, QFileDialog
from PyQt5.QtGui import QColor, QPainter, QPaintEvent, QImage, QWheelEvent, QMouseEvent
from PyQt5.QtCore import Qt, pyqtSignal, QSize, QPoint
QString = str
from .ui_control import UIControl

from data import lab_gamut, colorize_image
from skimage import color
import os
import datetime
import glob
import sys
import warnings

class GUIDraw(QWidget):
    update_color = pyqtSignal(QString)
    update_gamut = pyqtSignal(np.float64)
    suggest_colors = pyqtSignal(np.ndarray)
    used_colors = pyqtSignal(np.ndarray)
    update_ab = pyqtSignal(np.ndarray)
    update_result = pyqtSignal(np.ndarray)

    def __init__(
            self, 
            model: colorize_image.ColorizeImageTorch,
            dist_model: colorize_image.ColorizeImageTorchDist | None = None,
            load_size: int = 256,
            win_size: int = 512):
        QWidget.__init__(self)
        self.model = None
        self.image_file = None
        self.pos = None
        self.model = model
        self.dist_model = dist_model  # distribution predictor, could be empty
        self.win_size = win_size
        self.load_size = load_size
        self.setFixedSize(win_size, win_size)
        self.uiControl = UIControl(win_size=win_size, load_size=load_size)
        self.move(win_size, win_size)
        self.movie = True
        self.init_color()  # initialize color
        self.im_gray3 = None
        self.eraseMode = False
        self.ui_mode = 'none'   # stroke or point
        self.image_loaded = False
        self.use_gray = True
        self.total_images = 0
        self.image_id = 0
        self.method = 'with_dist'

    def clock_count(self) -> None:
        self.count_secs -= 1
        self.update()

    def init_result(self, image_file: str) -> None:
        self.read_image(image_file.encode('utf-8'))  # read an image
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
            print('you have finished all the results')
            sys.exit()
        img_current = self.img_list[self.image_id]
        # self.reset()
        self.init_result(img_current)
        self.reset_timer()

    def read_image(self, image_file: str) -> None:
        # self.result = None
        self.image_loaded = True
        self.image_file = image_file
        print(image_file)
        image_file = image_file.decode('utf8')
        im_bgr = cv2.imread(image_file)
        self.im_full = im_bgr.copy()
        # get image for display
        h, w, c = self.im_full.shape
        max_width = max(h, w)
        r = self.win_size / float(max_width)
        self.scale = float(self.win_size) / self.load_size
        print('scale = %f' % self.scale)
        rw = int(round(r * w / 4.0) * 4)
        rh = int(round(r * h / 4.0) * 4)

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
        lab_win = color.rgb2lab(self.im_win[:, :, ::-1])

        self.im_lab = color.rgb2lab(im_bgr[:, :, ::-1])
        self.im_l = self.im_lab[:, :, 0]
        self.l_win = lab_win[:, :, 0]
        self.im_ab = self.im_lab[:, :, 1:]
        self.im_size = self.im_rgb.shape[0:2]

        self.im_ab0 = np.zeros((2, self.load_size, self.load_size))
        self.im_mask0 = np.zeros((1, self.load_size, self.load_size))
        self.brushWidth = 2 * self.scale

        self.model.load_image(image_file)

        if (self.dist_model is not None):
            self.dist_model.set_image(self.im_rgb)
            self.predict_color()

    def update_im(self) -> None:
        self.update()
        QApplication.processEvents()

    def update_ui(self, move_point: bool = True) -> None:
        if self.ui_mode == 'none':
            return False
        is_predict = False
        snap_qcolor = self.calibrate_color(self.user_color, self.pos)
        self.color = snap_qcolor
        self.update_color.emit(QString('background-color: %s' % self.color.name()))

        if self.ui_mode == 'point':
            if move_point:
                self.uiControl.movePoint(self.pos, snap_qcolor, self.user_color, self.brushWidth)
            else:
                self.user_color, self.brushWidth, isNew = self.uiControl.addPoint(self.pos, snap_qcolor, self.user_color, self.brushWidth)
                if isNew:
                    is_predict = True
                    # self.predict_color()

        if self.ui_mode == 'stroke':
            self.uiControl.addStroke(self.prev_pos, self.pos, snap_qcolor, self.user_color, self.brushWidth)
        if self.ui_mode == 'erase':
            isRemoved = self.uiControl.erasePoint(self.pos)
            if isRemoved:
                is_predict = True
                # self.predict_color()
        return is_predict

    def reset(self) -> None:
        self.ui_mode = 'none'
        self.pos = None
        self.result = None
        self.user_color = None
        self.color = None
        self.uiControl.reset()
        self.init_color()
        self.compute_result()
        self.predict_color()
        self.update()

    def scale_point(self, pnt: QPoint) -> tuple[int, int]:
        x = int((pnt.x() - self.dw) / float(self.win_w) * self.load_size)
        y = int((pnt.y() - self.dh) / float(self.win_h) * self.load_size)
        return x, y

    def valid_point(self, pnt: QPoint) -> QPoint | None:
        if pnt is None:
            warnings.warn(f"'change_color()' got a 'pnt' of type 'None'.", RuntimeWarning)
            return None
        else:
            if pnt.x() >= self.dw and pnt.y() >= self.dh and pnt.x() < self.win_size - self.dw and pnt.y() < self.win_size - self.dh:
                x = int(np.round(pnt.x()))
                y = int(np.round(pnt.y()))
                return QPoint(x, y)
            else:
                warnings.warn(f"Point ({pnt.x()}, {pnt.y()}) out of bounds.", RuntimeWarning)
                return None

    def init_color(self) -> None:
        self.user_color = QColor(128, 128, 128)  # default color red
        self.color = self.user_color

    def change_color(self, pos: QPoint) -> None:
        if not isinstance(pos, QPoint):
            warnings.warn(f"'change_color()' got a 'pos' not of type 'QPoint'.", RuntimeWarning)
            return
        
        x, y = self.scale_point(pos)
        L = self.im_lab[y, x, 0]
        self.update_gamut.emit(L)
        rgb_colors = self.suggest_color(h=y, w=x, K=9)
        rgb_colors[-1, :] = 0.5

        self.suggest_colors.emit(rgb_colors)
        used_colors = self.uiControl.get_recently_used_colors()
        if used_colors is not None:
            self.used_colors.emit(used_colors)
        snap_color = self.calibrate_color(self.user_color, pos)
        c = np.array((snap_color.red(), snap_color.green(), snap_color.blue()), np.uint8)

        self.update_ab.emit(c)

    def calibrate_color(self, c: QColor, pos: QPoint) -> QColor:
        x, y = self.scale_point(pos)

        # snap color based on L color
        color_array = np.array((c.red(), c.green(), c.blue())).astype('uint8')
        mean_L = self.im_l[y, x]
        snap_color = lab_gamut.snap_ab(mean_L, color_array)
        snap_qcolor = QColor(snap_color[0], snap_color[1], snap_color[2])
        return snap_qcolor

    def set_color(self, c_rgb: np.ndarray) -> None:
        if self.pos is None:
            warnings.warn(f"'set_color()' got a 'c_rgb' of type 'None'.", RuntimeWarning)
            return
        c = QColor(c_rgb[0], c_rgb[1], c_rgb[2])
        self.user_color = c
        snap_qcolor = self.calibrate_color(c, self.pos)
        self.color = snap_qcolor
        self.update_color.emit(QString('background-color: %s' % self.color.name()))
        self.uiControl.update_color(snap_qcolor, self.user_color)
        self.compute_result()

    def erase(self) -> None:
        self.eraseMode = not self.eraseMode

    def load_image(self) -> None:
        img_path = QFileDialog.getOpenFileName(self, 'load an input image')[0]
        if os.path.isfile(img_path):
            self.init_result(img_path)

    def save_result(self) -> None:
        path = os.path.abspath(self.image_file)
        path, ext = os.path.splitext(path)

        suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
        path = path.decode('utf8')
        save_path = "_".join([path, self.method, suffix])

        print('saving result to <%s>\n' % save_path)
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        np.save(os.path.join(save_path, 'im_l.npy'), self.model.img_l)
        np.save(os.path.join(save_path, 'im_ab.npy'), self.im_ab0)
        np.save(os.path.join(save_path, 'im_mask.npy'), self.im_mask0)

        result_bgr = cv2.cvtColor(self.result, cv2.COLOR_RGB2BGR)
        mask = self.im_mask0.transpose((1, 2, 0)).astype(np.uint8) * 255
        cv2.imwrite(os.path.join(save_path, 'input_mask.png'), mask)
        cv2.imwrite(os.path.join(save_path, 'ours.png'), result_bgr)
        cv2.imwrite(os.path.join(save_path, 'ours_fullres.png'), self.model.get_img_fullres()[:, :, ::-1])
        cv2.imwrite(os.path.join(save_path, 'input_fullres.png'), self.model.get_input_img_fullres()[:, :, ::-1])
        cv2.imwrite(os.path.join(save_path, 'input.png'), self.model.get_input_img()[:, :, ::-1])
        cv2.imwrite(os.path.join(save_path, 'input_ab.png'), self.model.get_sup_img()[:, :, ::-1])

    def enable_gray(self) -> None:
        self.use_gray = not self.use_gray
        self.update()

    def predict_color(self) -> None:
        if self.dist_model is not None and self.image_loaded:
            im, mask = self.uiControl.get_input()
            im_mask0 = mask > 0.0
            self.im_mask0 = im_mask0.transpose((2, 0, 1))
            im_lab = color.rgb2lab(im).transpose((2, 0, 1))
            self.im_ab0 = im_lab[1:3, :, :]

            self.dist_model.net_forward(self.im_ab0, self.im_mask0)

    def suggest_color(self, h: int, w: int, K: int = 5) -> np.ndarray | None:
        if self.dist_model is not None and self.image_loaded:
            ab, conf = self.dist_model.get_ab_reccs(h=h, w=w, K=K, N=25000, return_conf=True)
            L = np.tile(self.im_lab[h, w, 0], (K, 1))
            colors_lab = np.concatenate((L, ab), axis=1)
            colors_lab3 = colors_lab[:, np.newaxis, :]
            colors_rgb = np.clip(np.squeeze(color.lab2rgb(colors_lab3)), 0, 1)
            colors_rgb_withcurr = np.concatenate((self.model.get_img_forward()[h, w, np.newaxis, :] / 255., colors_rgb), axis=0)
            return colors_rgb_withcurr
        else:
            warnings.warn(f"'suggest_color()' did not return a color suggestion.", RuntimeWarning)
            return None

    def compute_result(self) -> None:
        im, mask = self.uiControl.get_input()
        im_mask0 = mask > 0.0
        self.im_mask0 = im_mask0.transpose((2, 0, 1))
        im_lab = color.rgb2lab(im).transpose((2, 0, 1))
        self.im_ab0 = im_lab[1:3, :, :]

        self.model.net_forward(self.im_ab0, self.im_mask0)
        ab = self.model.output_ab.transpose((1, 2, 0))
        ab_win = cv2.resize(ab, (self.win_w, self.win_h), interpolation=cv2.INTER_CUBIC)
        pred_lab = np.concatenate((self.l_win[..., np.newaxis], ab_win), axis=2)
        pred_rgb = (np.clip(color.lab2rgb(pred_lab), 0, 1) * 255).astype('uint8')
        self.result = pred_rgb
        self.update_result.emit(self.result)
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
        print('update brushWidth = %f' % self.brushWidth)
        self.update_ui(move_point=True)
        self.update()

    def is_same_point(self, pos1: QPoint, pos2: QPoint) -> bool:
        if pos1 is None or pos2 is None:
            warnings.warn(f"'is_same_point()' attempted to compare 'None' type.", RuntimeWarning)
            return False
        dx = pos1.x() - pos2.x()
        dy = pos1.y() - pos2.y()
        d = dx * dx + dy * dy
        # print('distance between points = %f' % d)
        return d < 25

    def mousePressEvent(self, event: QMouseEvent) -> None:
        print('mouse press', event.pos())
        pos = self.valid_point(event.pos())

        if pos is not None:
            if event.button() == Qt.LeftButton:
                self.pos = pos
                self.ui_mode = 'point'
                self.change_color(pos)
                self.update_ui(move_point=False)
                self.compute_result()

            if event.button() == Qt.RightButton:
                # draw the stroke
                self.pos = pos
                self.ui_mode = 'erase'
                self.update_ui(move_point=False)
                self.compute_result()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        self.pos = self.valid_point(event.pos())
        if self.pos is not None:
            if self.ui_mode == 'point':
                self.update_ui(move_point=True)
                self.compute_result()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        pass

    def sizeHint(self) -> QSize:
        return QSize(self.win_size, self.win_size)  # 28 * 8
