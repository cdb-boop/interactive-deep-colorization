from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QPainter, QColor, QImage, QPaintEvent, QMouseEvent
from PyQt5.QtCore import Qt, pyqtSignal, QSize
import numpy as np


class GUIVisualize(QWidget):
    color_selected = pyqtSignal(np.ndarray)  # 1x3 uint8

    def __init__(self, win_size: int = 256, scale: float = 2.0):
        QWidget.__init__(self)
        self.image = None
        self.win_width = win_size
        self.win_height = win_size
        self.scale = scale
        self.setFixedSize(self.win_width, self.win_height)

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter()
        painter.begin(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(event.rect(), QColor(49, 54, 49))
        if self.image is not None:
            h, w, c = self.image.shape
            qImg = QImage(self.image.tobytes(), w, h, QImage.Format_RGB888)
            dw = int((self.win_width - w) // 2)
            dh = int((self.win_height - h) // 2)
            painter.drawImage(dw, dh, qImg)

        painter.end()

    def set_image(self, image: np.ndarray) -> None:
        self.image = image
        self.update()

    def sizeHint(self) -> QSize:
        return QSize(self.win_width, self.win_height)

    def reset(self) -> None:
        self.update()
        self.image = None

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if self.image is None:
            return

        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.pos()
            x = pos.x()
            y = pos.y()

            dim_y, dim_x = self.image.shape[0], self.image.shape[1]
            if dim_x != dim_y:
                if dim_x > dim_y:
                    y = y - ((dim_x - dim_y) // 2)
                else:
                    x = x - ((dim_y - dim_x) // 2)

            if x >= 0 and y >= 0 and x < dim_x and y < dim_y:
                self.color_selected.emit(self.image[y, x, :])

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        pass

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        pass
