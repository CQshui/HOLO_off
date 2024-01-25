import sys
from PIL import Image
import numpy as np
from PyQt5.QtGui import QIcon, QPixmap, QImage
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QAction, qApp, QFileDialog


class Window(QMainWindow):
    """Main Window."""

    def __init__(self, parent=None):
        """Initializer."""
        super().__init__(parent)
        self.label_w = None
        self.label_h = None
        self.label_show_camera = None
        # 菜单栏图标
        icon = QIcon(":/images/logo")
        self.setWindowIcon(icon)
        self.setWindowTitle("Test")
        # 初始化UI和文件路径
        self.initUI()
        self.filePaths = []
        self.showMaximized()

    def initUI(self):
        menubar = self.menuBar()
        # Menu退出
        exitAct = QAction(QIcon('./Resources/images/icon.ico'), '&Exit', self)
        exitAct.setShortcut('Ctrl+Q')
        exitAct.setStatusTip('退出应用')
        exitAct.triggered.connect(qApp.quit)
        fileMenu = menubar.addMenu('&File')
        # Menu导入图像
        impAct = QAction(QIcon('./Resources/images/icon.ico'), '&Import', self)
        impAct.setShortcut('Ctrl+N')
        impAct.setStatusTip('导入图像')
        impAct.triggered.connect(self.opening_pic)

        toolMenu = menubar.addMenu('Tool')

        # 功能添加至Menu
        fileMenu.addAction(impAct)
        fileMenu.addAction(exitAct)
        # 底部状态栏
        self.statusBar()

        # 框架搭建
        self.label_h = 800
        self.label_w = 800
        self.label_show_camera = QLabel(self)
        self.label_show_camera.move(600, 150)
        self.label_show_camera.setFixedSize(self.label_w, self.label_h)
        self.label_show_camera.setText("请导入图片")
        self.label_show_camera.setStyleSheet("QLabel{background:white;}")
        self.label_show_camera.setObjectName("image_show")

    def opening_pic(self):
        try:
            path1, _ = QFileDialog.getOpenFileName(self, "请选择文件", "", "All Files (*);;JPG Files (*.jpg)")
            self.filePaths.append(path1)
            # print(self.filePaths[-1])
            show = Image.open(self.filePaths[-1]).convert("RGB")
            show = show.resize([self.label_w, self.label_h])
            showImage = QImage(np.array(show), np.shape(show)[1], np.shape(show)[0], QImage.Format_RGB888)
            self.label_show_camera.setPixmap(QPixmap.fromImage(showImage))
            self.show()
        except:
            pass


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = Window()
    win.show()
    sys.exit(app.exec_())
