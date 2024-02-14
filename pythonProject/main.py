import sys
from PyQt6.QtWidgets import *
from PyQt6.QtGui import QPixmap
from PyQt6 import uic
import MnistNet

form_main = uic.loadUiType('main.ui')[0]

class WindowClass(QMainWindow, form_main):
    def __init__(self):
        super( ).__init__( )
        self.setupUi(self)
        self.setWindowTitle("MnistNet Test Program")
        self.btn_a.clicked.connect(self.fnc_a)
        self.mnistnet = MnistNet.Net()

    def fnc_a(self):
        filename = QFileDialog.getOpenFileName(self, 'Open File')
        filename = str(filename[0])

        # When you selected a file
        if len(filename) > 0:
            prediction = self.mnistnet.test_image(filename)
            self.label_number.setPixmap(QPixmap(filename))
            self.label_prediction.setText(str(prediction))




if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = WindowClass()
    window.show()
    app.exec( )