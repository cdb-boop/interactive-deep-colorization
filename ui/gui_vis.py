from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QPainter, QColor, QImage, QPaintEvent, QMouseEvent
from PyQt5.QtCore import Qt, pyqtSignal, QSize, QPoint
QString = str
import numpy as np
import warnings


class GUIVis(QWidget):
    update_color = pyqtSignal(QString)

    def __init__(self, win_size: int = 256, scale: float = 2.0):
        QWidget.__init__(self)
        self.result = None
        self.win_width = win_size
        self.win_height = win_size
        self.scale = scale
        self.setFixedSize(self.win_width, self.win_height)

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter()
        painter.begin(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(event.rect(), QColor(49, 54, 49))
        if self.result is not None:
            h, w, c = self.result.shape
            qImg = QImage(self.result.tostring(), w, h, QImage.Format_RGB888)
            dw = int((self.win_width - w) // 2)
            dh = int((self.win_height - h) // 2)
            painter.drawImage(dw, dh, qImg)

        painter.end()

    def update_result(self, result: np.ndarray) -> None:
        self.result = result
        self.update()

    def sizeHint(self) -> QSize:
        return QSize(self.win_width, self.win_height)

    def reset(self) -> None:
        self.update()
        self.result = None

    def is_valid_point(self, pos) -> bool:
        if not isinstance(pos, QPoint):
            warnings.warn("GUIVis: 'pos' was not 'QPoint'", RuntimeWarning)
            return False
        else:
            x = pos.x()
            y = pos.y()
            return x >= 0 and y >= 0 and x < self.win_width and y < self.win_height

    def scale_point(self, pnt: QPoint) -> tuple[int, int]:
        x = int(pnt.x() / self.scale)
        y = int(pnt.y() / self.scale)
        return x, y

    def mousePressEvent(self, event: QMouseEvent) -> None:
        pos = event.pos()
        x, y = self.scale_point(pos)
        if event.button() == Qt.LeftButton and self.is_valid_point(pos):  # click the point
            if self.result is not None:
                color = self.result[y, x, :]
                print(f"GUIVis: Color {color} at point ({pos.x()}, {pos.y()})")

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        pass

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        pass
