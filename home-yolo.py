# home-yolo.py
"""
Main script to run the License Plate Recognition (LPR) application. This application
uses deep learning models to detect and recognize license plates and characters.
It is built with PySide6 for the GUI and utilizes PyTorch for model inference.
Requirements:
- PySide6 for the GUI
- PyTorch for deep learning inference
- Pillow for image processing
- OpenCV for video and image manipulation
"""
import os
import functools
import gc
import statistics
import time
import warnings
import torch
import cv2
from jdatetime import date as jdate
import pandas as pd
from jalali_calendar_qt import JalaliDateEdit  
from datetime import datetime, timedelta
import jdatetime
from helper import jalali
from pathlib import Path
from PIL import ImageOps
from PySide6 import QtWidgets
from PySide6 import QtCore
from PySide6.QtCore import QThread, Signal, QSize, QDateTime, QTime, Qt
from PySide6.QtGui import QImage, QIcon, QAction, QPainter, QFont,  QPixmap
from PySide6.QtWidgets import QTableWidgetItem, QGraphicsScene, QMenu, QFileDialog, QInputDialog, QMessageBox, QDateTimeEdit, QLineEdit, QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTableWidget, QTimeEdit, QScrollArea
from PySide6.QtCharts import QChart, QPieSeries, QChartView, QLineSeries
from qtpy.uic import loadUi
import ai.img_model as imgModel
from ai.img_model import *
from configParams import Parameters
from database.db_entries_utils import *
from helper.gui_maker import configure_main_table_widget, create_image_label, on_label_double_click, center_widget
from helper.text_decorators import convert_english_to_persian, clean_license_plate_text, join_elements, \
    convert_persian_to_english, split_string_language_specific
import sys

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
params = Parameters()


sys.path.append('yolov5')


def get_device():
    """
    Determines the device to run the PyTorch models on.
    Returns a torch.device object representing the device (CUDA, MPS, or CPU).
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


modelPlate = torch.hub.load('yolov5', 'custom', params.modelPlate_path, source='local', force_reload=True)
modelCharX = torch.hub.load('yolov5', 'custom', params.modelCharX_path, source='local', force_reload=True)

device = get_device()  # تابع شما که CUDA, MPS یا CPU را تشخیص می‌دهد
modelPlate.to(device)
modelCharX.to(device)


class MainWindow(QtWidgets.QMainWindow):
    """
    The main window class of the LPR application.
    It sets up the user interface and connects signals and slots.
    """

    def __init__(self):
        """
       Initializes the main window and its components.
       """
        super(MainWindow, self).__init__()
        loadUi('./gui/mainFinal.ui', self)
        self.setFixedSize(self.size())


        self.camImage = None
        self.plateImage = None
        self.startButton.clicked.connect(self.start_webcam)
        self.stopButton.clicked.connect(self.stop_webcam)
        self.sourceButton.clicked.connect(self.select_source)
        self.searchBox.textChanged.connect(self.search_table)
        self.searchButton.clicked.connect(self.open_search_window)

        self.startButton.setIcon(QPixmap("./icons/play.png"))
        self.startButton.setIconSize(QSize(40, 40))

        self.stopButton.setIcon(QPixmap("./icons/stop.png"))
        self.stopButton.setIconSize(QSize(40, 40))

        self.sourceButton.setIcon(QPixmap("./icons/tools.png"))
        self.sourceButton.setIconSize(QSize(40, 40))

        self.searchButton.setIcon(QPixmap("./icons/search.png"))
        self.searchButton.setIconSize(QSize(60, 60))


        self.plateTextView.setStyleSheet(
            f"""border-image: url("{Path().absolute()}/Templates/template-base.png") 0 0 0 0 stretch stretch;""")

        self.Worker1 = Worker1()
        self.Worker1.plateDataUpdate.connect(self.on_plate_data_update)
        self.Worker1.mainViewUpdate.connect(self.on_main_view_update)


        self.Worker2 = Worker2()
        self.Worker2.mainTableUpdate.connect(self.refresh_table)
        self.Worker2.start()

        configure_main_table_widget(self)
        self.scene = QGraphicsScene()
        self.gv.setScene(self.scene)

        torch.cuda.empty_cache()
        gc.collect()

    def refresh_table(self, plateNum=''):
        # حذف limit برای نمایش همه ورودی‌ها
        plateNum = dbGetAllEntries(whereLike=plateNum)  # بدون limit
        self.tableWidget.setRowCount(len(plateNum))
        for index, entry in enumerate(plateNum):
            # تبدیل پلاک به فارسی
            plate_num_persian = convert_english_to_persian(entry['plateNum'], display=True)
            self.tableWidget.setItem(index, 0, QTableWidgetItem(plate_num_persian))
            self.tableWidget.setItem(index, 1, QTableWidgetItem(entry['eTime']))
            self.tableWidget.setItem(index, 2, QTableWidgetItem(jalali.Gregorian(entry['eDate']).persian_string()))
            # عکس پلاک
            Image = QImage()
            plate_pic_path = entry['platePic']  # استفاده از کلید platePic
            if os.path.exists(plate_pic_path):
                Image.load(plate_pic_path)
                QcroppedPlate = QPixmap.fromImage(Image)
                item = create_image_label(QcroppedPlate)
                item.mousePressEvent = functools.partial(on_label_double_click, source_object=item)
                self.tableWidget.setCellWidget(index, 3, item)
                self.tableWidget.setRowHeight(index, 44)
            else:
                self.tableWidget.setItem(index, 3, QTableWidgetItem("عکس موجود نیست"))
            # درصد اطمینان
            self.tableWidget.setItem(index, 4, QTableWidgetItem(str(entry['charPercent'])))
            self.tableWidget.setItem(index, 5, QTableWidgetItem(str(entry['platePercent'])))
            # اضافه کردن دکمه حذف
            delete_button = QtWidgets.QPushButton()
            delete_button.setFlat(True)
            delete_button.setStyleSheet("QPushButton { background-color: transparent; border: 0px }")
            delete_button.setIcon(QPixmap("./icons/delete.png"))
            delete_button.setIconSize(QSize(24, 24))
            delete_button.clicked.connect(functools.partial(self.delete_row, index=index, plateNum=entry['plateNum'], eTime=entry['eTime'], eDate=entry['eDate']))
            self.tableWidget.setCellWidget(index, 6, delete_button)


    def delete_row(self, index, plateNum, eTime, eDate):
        reply = QMessageBox.question(self, 'تأیید حذف', f'آیا مطمئن هستید که می‌خواهید ورودی پلاک {convert_english_to_persian(plateNum, display=True)} در تاریخ {jalali.Gregorian(eDate).persian_string()} ساعت {eTime} را حذف کنید؟',
                                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            dbRemoveEntry(plateNum, eTime, eDate)
            self.refresh_table()
            QMessageBox.information(self, "موفقیت", "ورودی با موفقیت حذف شد")




    def on_main_view_update(self, mainViewImage):

        qp = QPixmap.fromImage(mainViewImage)

        self.scene.addPixmap(qp)
        self.scene.setSceneRect(0, 0, 960, 540)
        self.gv.fitInView(self.scene.sceneRect())
        self.gv.setRenderHints(QPainter.Antialiasing)

    def on_plate_data_update(self, cropped_plate: QImage, plate_text: str, char_conf_avg: float,
                             plate_conf_avg: float) -> None:

        # Check if the plate text is 8 characters long and the character confidence is above 70
        if len(plate_text) == 8 and char_conf_avg >= 70:
            # Set the plate view to display the cropped plate
            self.plate_view.setScaledContents(True)
            self.plate_view.setPixmap(QPixmap.fromImage(cropped_plate))

            # Convert the plate text to Persian and set the text for the plate number and plate text in Persian
            plt_text_num = convert_english_to_persian(plate_text[:6], display=True)
            plt_text_ir = convert_english_to_persian(plate_text[6:], display=True)
            self.plate_text_num.setText(plt_text_num)
            self.plate_text_ir.setText(plt_text_ir)

            # Clean the plate text 
            plate_text_clean = clean_license_plate_text(plate_text)

            # Create data for send into services
            external_service_data = {
                'plate_number': plt_text_num + '-' + plt_text_ir,
                'image': cropped_plate
            }
            # Add the plate text, character confidence, plate confidence, cropped plate to the database
            db_entries_time(plate_text_clean, char_conf_avg, plate_conf_avg, cropped_plate,
                            external_service_data=external_service_data)
            self.Worker2.start()

    def start_webcam(self):
        if params.source is None:
            QMessageBox.warning(self, "خطا", "لطفاً ابتدا یک منبع انتخاب کنید")
            return
        if not self.Worker1.isRunning():
            self.Worker1.start()
        else:
            self.Worker1.unPause()

    def stop_webcam(self):
        self.Worker1.stop()


    def select_source(self):
        menu = QMenu(self)
        action_file = menu.addAction("فایل از سیستم")
        action_webcam = menu.addAction("وبکم")
        action_rtsp = menu.addAction("RTSP با URL")
        action = menu.exec(self.sourceButton.mapToGlobal(QtCore.QPoint(0, self.sourceButton.height())))
        if action == action_file:
            file_path, _ = QFileDialog.getOpenFileName(self, "انتخاب فایل ویدیو", "", "Video Files (*.mp4 *.avi)")
            if file_path:
                params.source = 'video'
                params.video = file_path
                params.webcam = None
                params.rtps = None
                self.Worker1.change_source(file_path)
                if self.Worker1.Capture.isOpened():
                    QMessageBox.information(self, "موفقیت", "منبع فایل ویدیو با موفقیت اضافه شد")
                    self.start_webcam()
                else:
                    QMessageBox.critical(self, "خطا", "نمی‌توان به منبع فایل متصل شد")
        elif action == action_webcam:
            params.source = 'webcam'
            params.webcam = 0  # ایندکس پیش‌فرض
            params.video = None
            params.rtps = None
            self.Worker1.change_source(0)
            if self.Worker1.Capture.isOpened():
                # تست خواندن یک فریم
                if self.Worker1.Capture.isOpened():
                    QMessageBox.information(self, "موفقیت", "منبع وبکم با موفقیت اضافه شد")
                    self.start_webcam()  # Worker را اینجا اجرا کنید
                else:
                    QMessageBox.critical(self, "خطا", "نمی‌توان به وبکم متصل شد")


            else:
                QMessageBox.critical(self, "خطا", "نمی‌توان به وبکم متصل شد")
        elif action == action_rtsp:
            rtsp_url, ok = QInputDialog.getText(self, "RTSP URL", "آدرس RTSP را وارد کنید:")
            if ok and rtsp_url:
                params.source = 'rtps'
                params.rtps = rtsp_url
                params.webcam = None
                params.video = None
                self.Worker1.change_source(rtsp_url)
                if self.Worker1.Capture.isOpened():
                    QMessageBox.information(self, "موفقیت", "منبع RTSP با موفقیت اضافه شد")
                    self.start_webcam()
                else:
                    QMessageBox.critical(self, "خطا", "نمی‌توان به منبع RTSP متصل شد")

 

    def search_table(self):
        search_text = self.searchBox.toPlainText()
        self.refresh_table(plateNum=search_text)  # جستجو با whereLike


    def open_search_window(self):
        search_window = SearchWindow()
        center_widget(search_window)
        search_window.exec()

class Worker1(QThread):
    """
    Worker thread that handles frame grabbing and processing in the background.
    It is responsible for detecting plates and recognizing characters.
    """
    mainViewUpdate = Signal(QImage)
    plateDataUpdate = Signal(QImage, list, int, int)
    TotalFramePass = 0

    def __init__(self, parent=None):
        super().__init__(parent)
        self.Capture = None  # مقداردهی اولیه

    def run(self):
        # هر فریم
        self.prepare_capture()
        while self.ThreadActive:
            success, frame = self.Capture.read()
            if success:
                self.process_frame(frame)
                self.manageFrameRate()


    def prepare_capture(self):
        self.prev_frame_time = 0
        self.ThreadActive = True
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp|timeout;5000"
        source = None
        if params.source == 'webcam' and params.webcam is not None:
            source = params.webcam
        elif params.source == 'video' and params.video is not None:
            source = params.video
        elif params.source == 'rtps' and params.rtps is not None:
            source = params.rtps
        if source is None:
            self.ThreadActive = False
            return
        self.Capture = cv2.VideoCapture(source)
        if not self.Capture.isOpened():
            self.ThreadActive = False
            return
        self.adjust_video_position()

    def adjust_video_position(self):
        if params.source == 'video':
            total = int(self.Capture.get(cv2.CAP_PROP_FRAME_COUNT))
            self.TotalFramePass = 0 if self.TotalFramePass > total else self.TotalFramePass
            self.Capture.set(1, self.TotalFramePass)

    def process_frame(self, frame):
        self.TotalFramePass += 1
        resize = self.prepareImage(frame)

        platesResult = modelPlate(resize).pandas().xyxy[0]
        for _, plate in platesResult.iterrows():
            plateConf = int(plate['confidence'] * 100)
            if plateConf >= 60:
                self.highlightPlate(resize, plate)
                croppedPlate = self.cropPlate(resize, plate)
                plateText, char_detected, charConfAvg = self.detectPlateChars(croppedPlate)
                self.emitPlateData(croppedPlate, plateText, char_detected, charConfAvg, plateConf)

        self.emitFrame(resize)

    def prepareImage(self, frame):
        resize = cv2.resize(frame, (960, 540))
        effect = ImageOps.autocontrast(imgModel.to_img_pil(resize), cutoff=1)
        return cv2.cvtColor(imgModel.to_img_opencv(effect), cv2.COLOR_BGR2RGB)

    def highlightPlate(self, resize, plate):
        cv2.rectangle(resize, (int(plate['xmin']) - 3, int(plate['ymin']) - 3),
                      (int(plate['xmax']) + 3, int(plate['ymax']) + 3),
                      color=(0, 0, 255), thickness=3)

    def cropPlate(self, resize, plate):
        return resize[int(plate['ymin']): int(plate['ymax']), int(plate['xmin']): int(plate['xmax'])]

    def emitPlateData(self, croppedPlate, plateText, char_detected, charConfAvg, plateConf):
        croppedPlate = cv2.resize(croppedPlate, (600, 132))
        croppedPlateImage = QImage(croppedPlate.data, croppedPlate.shape[1], croppedPlate.shape[0],
                                   QImage.Format_RGB888)
        self.plateDataUpdate.emit(croppedPlateImage, plateText, charConfAvg, plateConf)

    def manageFrameRate(self):
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - self.prev_frame_time)
        self.prev_frame_time = new_frame_time
        self.currentFPS = fps  # Save the current FPS for later drawing on the frame

    def emitFrame(self, resize):
        if hasattr(self, 'currentFPS'):  # Check if currentFPS has been calculated
            imgModel.draw_fps(resize, self.currentFPS)  # نمایش fps
        mainFrame = QImage(resize.data, resize.shape[1], resize.shape[0], QImage.Format_RGB888)
        self.mainViewUpdate.emit(mainFrame)

    def detectPlateChars(self, croppedPlate):
        chars, confidences, char_detected = [], [], []
        results = modelCharX(croppedPlate)
        detections = results.pred[0]
        detections = sorted(detections, key=lambda x: x[0])  # sort by x coordinate
        for det in detections:
            conf = det[4]
            if conf > 0.5:
                cls = det[5].item()
                char = params.char_id_dict.get(str(int(cls)), '')
                chars.append(char)
                confidences.append(conf.item())
                char_detected.append(det.tolist())
        charConfAvg = round(statistics.mean(confidences) * 100) if confidences else 0
        return ''.join(chars), char_detected, charConfAvg
        

    def unPause(self):
        self.ThreadActive = True

    def stop(self):
        self.ThreadActive = False

    def change_source(self, source):
        if hasattr(self, 'Capture') and self.Capture is not None and self.Capture.isOpened():
            self.Capture.release()
        self.Capture = cv2.VideoCapture(source)
        time.sleep(1)  # تأخیر برای اطمینان از باز شدن منبع
        self.adjust_video_position()


class Worker2(QThread):
    mainTableUpdate = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

    def run(self):
        self.mainTableUpdate.emit()
        time.sleep(.5)

    def unPause(self):
        self.ThreadActive = True

    def stop(self):
        self.ThreadActive = False


class SearchWindow(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("جستجو بر اساس بازه زمانی")
        self.setFixedSize(1280, 720)

        layout = QVBoxLayout()

        self.setLayoutDirection(Qt.RightToLeft)

        # ورودی‌ها
        input_layout = QHBoxLayout()

        self.start_date = JalaliDateEdit(self)
        yesterday = jdatetime.date.today() - timedelta(days=1)
        self.start_date.set_date(yesterday)  # پیش‌فرض یک روز قبل
        input_layout.addWidget(QLabel("تاریخ شروع (شمسی):"))
        input_layout.addWidget(self.start_date)

        self.start_time = QTimeEdit(self)
        self.start_time.setTime(QTime(0, 0, 0))
        self.start_time.setLayoutDirection(Qt.LeftToRight)
        self.start_time.setAlignment(Qt.AlignLeft)
        self.start_time.setDisplayFormat("HH:mm:ss")
        input_layout.addWidget(QLabel("ساعت شروع:"))
        input_layout.addWidget(self.start_time)

        self.end_date = JalaliDateEdit(self)
        self.end_date.set_date(jdatetime.date.today())  # پیش‌فرض امروز
        input_layout.addWidget(QLabel("تاریخ پایان (شمسی):"))
        input_layout.addWidget(self.end_date)

        self.end_time = QTimeEdit(self)
        self.end_time.setTime(QTime(23, 59, 59))
        self.end_time.setLayoutDirection(Qt.LeftToRight)
        self.end_time.setAlignment(Qt.AlignLeft)
        self.end_time.setDisplayFormat("HH:mm:ss")
        input_layout.addWidget(QLabel("ساعت پایان:"))
        input_layout.addWidget(self.end_time)

        self.plate_filter = QLineEdit(self)
        self.plate_filter.setPlaceholderText("فیلتر پلاک (اختیاری)")
        input_layout.addWidget(QLabel("فیلتر پلاک:"))
        input_layout.addWidget(self.plate_filter)

        search_btn = QPushButton("جستجو")
        search_btn.clicked.connect(self.perform_search)
        input_layout.addWidget(search_btn)

        layout.addLayout(input_layout)

        # جدول نتایج
        self.tableWidget = QTableWidget()
        configure_main_table_widget(self)
        layout.addWidget(self.tableWidget)

        # چیدمان افقی برای نمودار و آمار
        chart_stats_layout = QHBoxLayout()
        
        # نمودار
        self.chart_view = QChartView()
        self.chart_view.setMaximumSize(400, 300)  # محدود کردن اندازه نمودار
        chart_stats_layout.addWidget(self.chart_view)

        # آمار
        self.stats_label = QLabel()
        self.stats_label.setWordWrap(True)
        stats_scroll = QScrollArea()
        stats_scroll.setWidget(self.stats_label)
        stats_scroll.setWidgetResizable(True)
        stats_scroll.setMaximumHeight(150)
        chart_stats_layout.addWidget(stats_scroll)

        layout.addLayout(chart_stats_layout)

        # دکمه خروجی‌ها (اکسل + متن) در یک ردیف
        buttons_layout = QHBoxLayout()

        export_btn = QPushButton("📊 خروجی به اکسل")
        export_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                border-radius: 8px;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)

        export_txt_btn = QPushButton("📝 خروجی به متن")
        export_txt_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                font-weight: bold;
                border-radius: 8px;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)

        export_btn.clicked.connect(self.export_to_excel)
        export_txt_btn.clicked.connect(self.export_to_txt)

        buttons_layout.addWidget(export_btn)
        buttons_layout.addWidget(export_txt_btn)

        layout.addLayout(buttons_layout)


        self.setLayout(layout)


    def perform_search(self):
        start_persian_date = self.start_date.date()  # JalaliDate
        start_time_str = self.start_time.time().toString("hh:mm:ss")
        end_persian_date = self.end_date.date()
        end_time_str = self.end_time.time().toString("hh:mm:ss")
        plate_filter = self.plate_filter.text()

        # تبدیل به میلادی
        try:
            start_gregorian = start_persian_date.to_gregorian().strftime("%Y-%m-%d")
        except AttributeError:
            try:
                start_gregorian = start_persian_date.togregorian().strftime("%Y-%m-%d")
            except AttributeError:
                start_gregorian = start_persian_date.strftime("%Y-%m-%d")

        try:
            end_gregorian = end_persian_date.to_gregorian().strftime("%Y-%m-%d")
        except AttributeError:
            try:
                end_gregorian = end_persian_date.togregorian().strftime("%Y-%m-%d")
            except AttributeError:
                end_gregorian = end_persian_date.strftime("%Y-%m-%d")


        start_datetime = f"{start_gregorian} {start_time_str}"
        end_datetime = f"{end_gregorian} {end_time_str}"

        entries = dbGetEntriesByDateTime(start_datetime, end_datetime, plate_filter)

        if not entries:
            self.tableWidget.setRowCount(0)  # جدول رو خالی کن
            # آمار رو خالی کن
            self.stats_label.setText("")

            # نمودار رو خالی کن
            empty_chart = QChart()
            self.chart_view.setChart(empty_chart)

            QMessageBox.information(self, "نتیجه", "هیچ ورودی در این بازه پیدا نشد")
            return

        self.tableWidget.setRowCount(len(entries))
        for i, entry in enumerate(entries):
            
            # تبدیل پلاک به فارسی
            plate_num_persian = convert_english_to_persian(entry['plateNum'], display=True)
            self.tableWidget.setItem(i, 0, QTableWidgetItem(plate_num_persian))
            self.tableWidget.setItem(i, 1, QTableWidgetItem(entry['eTime']))
            self.tableWidget.setItem(i, 2, QTableWidgetItem(jalali.Gregorian(entry['eDate']).persian_string()))

            # عکس پلاک
            image = QImage()
            plate_pic_path = entry['platePic']  # بدون جایگزینی : با -
            if os.path.exists(plate_pic_path):
                image.load(plate_pic_path)
                q_cropped_plate = QPixmap.fromImage(image)
                item = create_image_label(q_cropped_plate)
                item.mousePressEvent = functools.partial(on_label_double_click, source_object=item)
                self.tableWidget.setCellWidget(i, 3, item)
                self.tableWidget.setRowHeight(i, 44)
            else:
                self.tableWidget.setItem(i, 3, QTableWidgetItem("عکس موجود نیست"))

            # درصد اطمینان
            self.tableWidget.setItem(i, 4, QTableWidgetItem(str(entry['charPercent'])))
            self.tableWidget.setItem(i, 5, QTableWidgetItem(str(entry['platePercent'])))
            # اضافه کردن دکمه حذف
            delete_button = QtWidgets.QPushButton()
            delete_button.setFlat(True)
            delete_button.setStyleSheet("QPushButton { background-color: transparent; border: 0px }")
            delete_button.setIcon(QPixmap("./icons/delete.png"))
            delete_button.setIconSize(QSize(24, 24))
            delete_button.clicked.connect(functools.partial(self.delete_row, index=i, plateNum=entry['plateNum'], eTime=entry['eTime'], eDate=entry['eDate']))
            self.tableWidget.setCellWidget(i, 6, delete_button)


        # محاسبه آمار
        total_crossings = len(entries)
        plate_counts = {}
        for entry in entries:
            plate = entry['plateNum']
            plate_counts[plate] = plate_counts.get(plate, 0) + 1

        max_crossings = max(plate_counts, key=plate_counts.get) if plate_counts else None
        stats_text = f"تعداد کل عبور: {total_crossings}\n"
        if max_crossings:
            max_crossings_persian = convert_english_to_persian(max_crossings, display=True)
            stats_text += f"بیشترین عبور: {max_crossings_persian} ({plate_counts[max_crossings]} بار)\n"
            stats_text += "10 پلاک پرتکرار:\n"
            sorted_plates = sorted(plate_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            for plate, count in sorted_plates:
                plate_persian = convert_english_to_persian(plate, display=True)
                stats_text += f"{plate_persian}: {count} بار\n"

        self.stats_label.setText(stats_text)

        # نمودار Pie برای بیشترین عبورها
        series = QPieSeries()
        for plate, count in sorted_plates:  # Top 10
            plate_persian = convert_english_to_persian(plate, display=True)
            series.append(plate_persian, count)

        chart = QChart()
        chart.addSeries(series)
        chart.setTitle("بیشترین عبورها (10 پلاک برتر)")
        chart.setTitleFont(QFont("Arial", 10))  # فونت کوچکتر برای عنوان
        chart.setMargins(QtCore.QMargins(10, 10, 10, 10))  # کاهش حاشیه‌ها
        chart.legend().setFont(QFont("Arial", 8))  # فونت کوچکتر برای افسانه
        chart.legend().setAlignment(QtCore.Qt.AlignRight)  # قرار دادن افسانه در سمت راست

        self.chart_view.setChart(chart)


    def delete_row(self, index, plateNum, eTime, eDate):
            reply = QMessageBox.question(self, 'تأیید حذف', f'آیا مطمئن هستید که می‌خواهید ورودی پلاک {convert_english_to_persian(plateNum, display=True)} در تاریخ {jalali.Gregorian(eDate).persian_string()} ساعت {eTime} را حذف کنید؟',
                                        QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                dbRemoveEntry(plateNum, eTime, eDate)
                self.perform_search()
                QMessageBox.information(self, "موفقیت", "ورودی با موفقیت حذف شد")


    def export_to_excel(self):
        data = []
        for row in range(self.tableWidget.rowCount()):
            row_data = []
            for column in [0, 1, 2, 4, 5]:  # ستون‌های پلاک، ساعت، تاریخ، درصد حروف، درصد پلاک
                item = self.tableWidget.item(row, column)
                row_data.append(item.text() if item else '')
            data.append(row_data)

        df = pd.DataFrame(data, columns=["پلاک خودرو", "ساعت", "تاریخ", "درصد حروف", "درصد پلاک"])
        file_path, _ = QFileDialog.getSaveFileName(self, "ذخیره اکسل", "", "Excel Files (*.xlsx)")
        if file_path:
            # اگر کاربر پسوند ننوشت، ما اضافه کنیم
            if not file_path.lower().endswith(".xlsx"):
                file_path += ".xlsx"

            df.to_excel(file_path, index=False)
            QMessageBox.information(self, "موفقیت", "داده‌ها به اکسل خروجی گرفته شد")

    def export_to_txt(self):
        data = []
        for row in range(self.tableWidget.rowCount()):
            row_data = []
            for column in [0, 1, 2, 4, 5]:  # ستون‌های مورد نظر
                item = self.tableWidget.item(row, column)
                row_data.append(item.text() if item else '')
            data.append("\t".join(row_data))  # ستون‌ها با tab جدا بشن

        file_path, _ = QFileDialog.getSaveFileName(self, "ذخیره فایل متنی", "", "Text Files (*.txt)")
        if file_path:
            # اگر کاربر پسوند ننوشت، خودمون اضافه کنیم
            if not file_path.lower().endswith(".txt"):
                file_path += ".txt"
            with open(file_path, "w", encoding="utf-8") as f:
                f.write("\n".join(data))
            QMessageBox.information(self, "موفقیت", "داده‌ها به فایل متنی خروجی گرفته شد")


def get_platform():
    platforms = {
        'linux1': 'Linux',
        'linux2': 'Linux',
        'darwin': 'OS X',
        'win32': 'Windows'
    }
    if sys.platform not in platforms:
        return sys.platform

    return platforms[sys.platform]

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Windows')
    window = MainWindow()
    window.setWindowIcon(QIcon("./icons/car.png"))
    window.setIconSize(QSize(16, 16))
    center_widget(window)
    window.show()
    sys.exit(app.exec())
