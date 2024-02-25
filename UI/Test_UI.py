import sys

from PIL import Image
from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt5.QtGui import QIcon
from PyQt6 import QtCore, QtGui, QtWidgets
import numpy as np
import cv2


class Ui_MainWindow(QMainWindow):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.showMaximized()
        # 重建参数设置区域
        self.centralwidget = QtWidgets.QWidget(parent=MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.groupBox = QtWidgets.QGroupBox(parent=self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(40, 40, 241, 91))
        self.groupBox.setFlat(False)
        self.groupBox.setObjectName("groupBox")
        self.label = QtWidgets.QLabel(parent=self.groupBox)
        self.label.setGeometry(QtCore.QRect(20, 30, 54, 16))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(parent=self.groupBox)
        self.label_2.setGeometry(QtCore.QRect(20, 60, 81, 16))
        self.label_2.setObjectName("label_2")
        self.lineEdit = QtWidgets.QLineEdit(parent=self.groupBox)
        self.lineEdit.setGeometry(QtCore.QRect(100, 30, 113, 20))
        self.lineEdit.setClearButtonEnabled(False)
        self.lineEdit.setObjectName("lineEdit")
        self.lineEdit_2 = QtWidgets.QLineEdit(parent=self.groupBox)
        self.lineEdit_2.setGeometry(QtCore.QRect(100, 60, 113, 20))
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.groupBox_3 = QtWidgets.QGroupBox(parent=self.centralwidget)
        self.groupBox_3.setGeometry(QtCore.QRect(40, 280, 241, 71))
        self.groupBox_3.setFlat(False)
        self.groupBox_3.setObjectName("groupBox_3")
        self.comboBox = QtWidgets.QComboBox(parent=self.groupBox_3)
        self.comboBox.setGeometry(QtCore.QRect(10, 30, 111, 22))
        self.comboBox.setFrame(True)
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.pushButton = QtWidgets.QPushButton(parent=self.groupBox_3)
        self.pushButton.setGeometry(QtCore.QRect(150, 30, 75, 24))
        self.pushButton.setObjectName("pushButton")
        self.groupBox_2 = QtWidgets.QGroupBox(parent=self.centralwidget)
        self.groupBox_2.setGeometry(QtCore.QRect(40, 160, 241, 91))
        self.groupBox_2.setFlat(False)
        self.groupBox_2.setObjectName("groupBox_2")
        self.label_3 = QtWidgets.QLabel(parent=self.groupBox_2)
        self.label_3.setGeometry(QtCore.QRect(20, 30, 61, 16))
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(parent=self.groupBox_2)
        self.label_4.setGeometry(QtCore.QRect(20, 60, 81, 16))
        self.label_4.setObjectName("label_4")
        self.lineEdit_3 = QtWidgets.QLineEdit(parent=self.groupBox_2)
        self.lineEdit_3.setGeometry(QtCore.QRect(100, 30, 41, 20))
        self.lineEdit_3.setObjectName("lineEdit_3")
        self.lineEdit_4 = QtWidgets.QLineEdit(parent=self.groupBox_2)
        self.lineEdit_4.setGeometry(QtCore.QRect(100, 60, 113, 20))
        self.lineEdit_4.setObjectName("lineEdit_4")
        self.lineEdit_5 = QtWidgets.QLineEdit(parent=self.groupBox_2)
        self.lineEdit_5.setGeometry(QtCore.QRect(170, 30, 41, 20))
        self.lineEdit_5.setObjectName("lineEdit_5")
        self.label_5 = QtWidgets.QLabel(parent=self.groupBox_2)
        self.label_5.setGeometry(QtCore.QRect(150, 30, 16, 16))
        self.label_5.setObjectName("label_5")
        # 图片显示区域
        self.ShowImage = QtWidgets.QScrollArea(parent=self.centralwidget)
        self.ShowImage.setGeometry(QtCore.QRect(330, 50, 381, 411))
        self.ShowImage.setFixedSize(600, 600)
        self.ShowImage.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.ShowImage.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.ShowImage.setObjectName("ShowImage")
        MainWindow.setCentralWidget(self.centralwidget)
        # 菜单栏区域
        self.menubar = QtWidgets.QMenuBar(parent=MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 22))
        self.menubar.setObjectName("menubar")
        self.menu = QtWidgets.QMenu(parent=self.menubar)
        self.menu.setObjectName("menu")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(parent=MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actiondakaitupian = QtGui.QAction(parent=MainWindow)
        self.actiondakaitupian.setObjectName("actiondakaitupian")
        self.actiondakaitupian.triggered.connect(self.opening_pic)
        self.menu.addAction(self.actiondakaitupian)
        self.menubar.addAction(self.menu.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.groupBox.setTitle(_translate("MainWindow", "拍摄参数"))
        self.label.setText(_translate("MainWindow", "波长(nm)"))
        self.label_2.setText(_translate("MainWindow", "像素尺寸(μm)"))
        self.groupBox_3.setTitle(_translate("MainWindow", "重建算法"))
        self.comboBox.setItemText(0, _translate("MainWindow", "菲涅尔重建"))
        self.pushButton.setText(_translate("MainWindow", "开始重建"))
        self.groupBox_2.setTitle(_translate("MainWindow", "重建范围"))
        self.label_3.setText(_translate("MainWindow", "Z范围(cm)"))
        self.label_4.setText(_translate("MainWindow", "Z间隔(mm)"))
        self.label_5.setText(_translate("MainWindow", "~"))
        self.menu.setTitle(_translate("MainWindow", "文件"))
        self.actiondakaitupian.setText(_translate("MainWindow", "打开图片"))

    def __init__(self):
        super(Ui_MainWindow, self).__init__()
        # self.setWindowFlag(QtCore.Qt.WindowType.MSWindowsFixedSizeDialogHint)
        self.setupUi(self)
        self.filePaths = []

    def opening_pic(self):
        try:
            path1, _ = QFileDialog.getOpenFileName(self, "请选择文件", "", "IMG Files (*.png *.jpg *.bmp)")
            self.filePaths.append(path1)
            # print(self.filePaths[-1])
            show = Image.open(self.filePaths[-1]).convert("RGB")
            show = show.resize([self.ShowImage.width(), self.ShowImage.height()])
            showImage = QImage(np.array(show), np.shape(show)[1], np.shape(show)[0], QImage.Format.Format_RGB888)
            self.ShowImage.setPixmap(QPixmap.fromImage(showImage))
            self.show()
        except Exception as e:
            print(e)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = Ui_MainWindow()
    win.show()
    sys.exit(app.exec())

