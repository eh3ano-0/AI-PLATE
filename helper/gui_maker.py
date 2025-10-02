# gui_maker.py

import functools
from PySide6 import QtWidgets, QtCore, QtGui
from PySide6.QtCore import QSize
from PySide6.QtGui import QColor, QImage, QPixmap, Qt, QScreen
from PySide6.QtWidgets import QLabel, QTableWidgetItem, QAbstractItemView, QVBoxLayout, QDialog, QApplication, \
    QTableWidget

from helper import jalali
from helper.text_decorators import *

class CenterAlignDelegate(QtWidgets.QStyledItemDelegate):
    """
    Custom delegate for aligning table items to the center.
    """

    def initStyleOption(self, option, index):
        super(CenterAlignDelegate, self).initStyleOption(option, index)
        option.displayAlignment = QtCore.Qt.AlignCenter



def create_image_label(image):
    """
    Creates a QLabel with a given image.

    Parameters:
    - image (QPixmap): Image to display on the label.

    Returns:
    - QLabel: A label widget displaying the given image.
    """
    imageLabel = QLabel()
    imageLabel.setText("")
    imageLabel.setScaledContents(True)
    imageLabel.setFixedSize(200, 44)
    imageLabel.setPixmap(image)
    return imageLabel


def configure_main_table_widget(self):
    """
       Configures the main table widget.
       """
    fieldsList = ['پلاک خودرو', 'ساعت', 'تاریخ', 'عکس پلاک', 'درصد اطمینان کاراکتر', 'درصد تشخیص ناحیه پلاک', 'حذف' ]

    self.tableWidget.setColumnCount(len(fieldsList))
    self.tableWidget.setRowCount(20)
    self.tableWidget.setHorizontalHeaderLabels(fieldsList)
    self.tableWidget.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
    self.tableWidget.setLayoutDirection(Qt.RightToLeft)

    delegate = CenterAlignDelegate(self.tableWidget)
    self.tableWidget.setItemDelegate(delegate)
    self.tableWidget.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
    self.tableWidget.setSelectionMode(QAbstractItemView.SingleSelection)
    self.tableWidget.setSelectionBehavior(QAbstractItemView.SelectRows)



def center_widget(wid):
    """
       Centers a widget on the screen.

       Parameters:
       - wid (QWidget): The widget to be centered.
       """
    center = QScreen.availableGeometry(QApplication.primaryScreen()).center()
    geo = wid.frameGeometry()
    geo.moveCenter(center)
    wid.move(geo.topLeft())


def on_label_double_click(event, source_object=None):
    """
       Handles double-click event on label to show image in a dialog.

       Parameters:
       - event: The mouse event.
       - source_object: The source label object containing the pixmap.
       """
    w = QDialog()
    w.setFixedSize(600, 132)
    w.setWindowTitle("نمایش پلاک")

    imageLabel = QLabel(w)
    imageLabel.setText("")
    imageLabel.setScaledContents(True)
    imageLabel.setFixedSize(600, 132)
    imageLabel.setPixmap(source_object.pixmap())

    layout = QVBoxLayout()
    layout.addWidget(imageLabel)
    w.exec()

