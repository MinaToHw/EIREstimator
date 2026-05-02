#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# @Author: 
# @Date:   2026/3/24
# @Last Modified by:   
# @Last Modified time: 14:58
# ----------------------------------------------------------------------------
import os
import sys
import io
import csv
import copy
import time
import inspect
import numpy as np
import matplotlib as mpl
import tensorflow as tf
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from scipy.signal import decimate, firwin, filtfilt
from ui.ui_Networks_Tester import Ui_Form as Tester
from ui.ui_channel_choose import Ui_Channel_choose
from ui.ui_signal_preprocess import Ui_Signal_preprocess
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar


def msg_cri(s):
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Critical)
    msg.setText(s)
    msg.setWindowTitle(" ")
    msg.setStandardButtons(QMessageBox.Ok)
    msg.exec_()


def msg_prompt(s):
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Question)
    msg.setText(s)
    msg.setWindowTitle(" ")
    msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
    result = msg.exec_()
    return result


class LineEdit(QLineEdit):
    KEY = Qt.Key_Return  # 定义一个类属性 KEY，其值为 Qt.Key_Return，即回车键

    def __init__(self, *args, **kwargs):
        QLineEdit.__init__(self, *args, **kwargs)  # 调用父类 QLineEdit 的构造函数，确保正确初始化
        QREV = QRegExpValidator(QRegExp("[+-]?\\d*[\\.]?\\d+"))  # 允许输入正负号、可选的小数点和数字，表示有效的浮点数格式
        QREV.setLocale(QLocale(QLocale.English))  # 设置验证器的区域为 QLocale(QLocale.English)，以确保小数点符号为点（.）
        self.setValidator(QREV)


class NN_Window(QtWidgets.QWidget, Tester):
    def __init__(self, parent=None):
        super(NN_Window, self).__init__()
        self.setupUi(self)
        self.updateWidget = None
        self.signal_preprocess_window = None
        self.Button_Run.clicked.connect(self.Runclick)
        self.Button_LoadNN.clicked.connect(self.LoadNetwork)
        self.Button_LoadData.clicked.connect(self.LoadData)
        self.color = QColor(255, 255, 255)
        self.Fix_Params = {}
        self.Free_Params = {}
        self.loaded_data = []
        self.preprocessed_signal = []

        self.NN_Signal_Window = NN_SignalWindow(self)
        self.verticalLayout.addWidget(self.NN_Signal_Window)

        self.network_info = []
        self.network_info.append("""
                            <style>
                                .notice { font-size: 10pt; font-weight: bold; margin: 5px 0; }
                                .training-epoch-detail { font-size: 8pt; margin-left: 5px; line-height: 1.4; }
                                .section-title { font-size: 13pt; font-weight: bold; margin: 10px 0 5px 0; }
                                .normal-text { font-size: 10pt; }
                            </style>
                            """)

    def LoadData(self):
        fileName = QFileDialog.getOpenFileName(self, caption='Load Data', filter="csv (*.csv);;mat (*.mat)")
        if fileName[0] == '':
            return
        if fileName[1] == 'csv (*.csv)':
            with open(fileName[0], mode='r') as csv_file:
                self.reader = csv.DictReader(csv_file)
                self.fieldnames = self.reader.fieldnames

                self.channel_choose_window = Channel_choose_Window(self)
                # self.signal_preprocess_window.exec_()
                if self.channel_choose_window.exec_() == QDialog.Accepted:
                    self.signal_preprocess_window = Signal_preprocess_Window(self)
                    self.signal_preprocess_window.exec_()
                signal_subject = self.preprocessed_signal[0]
                self.signal = signal_subject.dictionnaire['Data']
                self.Fs = signal_subject.dictionnaire['Freq']
                self.t = np.arange(0, len(self.signal)*(1 / signal_subject.dictionnaire['Freq']), 1 / signal_subject.dictionnaire['Freq'])
                self.NN_Signal_Window.updatesignal(self.t, self.signal)

    def LoadNetwork(self):
        fileName = QFileDialog.getExistingDirectory(self, "Open Network File", "")
        if fileName == '':
            return
        fileName = str(fileName)

        self.network_location = str(fileName)
        self.network = tf.keras.models.load_model(self.network_location)

        model_info = self.get_detailed_model_info()
        self.display_information(model_info, 'notice')
        return

    def get_detailed_model_info(self):
        """获取详细的模型信息，递归遍历所有层"""
        info_lines = []
        info_lines.append("\n" + "=" * 20)
        info_lines.append("Network Details")

        # 辅助函数来递归遍历模型
        def process_layer(layer, prefix=''):
            layer_info = []
            layer_info.append("\n" + "=" * 20)
            layer_info.append(f"\n{prefix}Layer: {layer.name}")
            layer_info.append(f"{prefix}  Type: {layer.__class__.__name__}")
            layer_info.append(f"{prefix}  Parameters: {layer.count_params():,}")

            details = []
            if hasattr(layer, 'filters'):
                details.append(f"{prefix} Filters: {layer.filters}")
            if hasattr(layer, 'kernel_size'):
                details.append(f"{prefix} Kernel Size: {layer.kernel_size}")
            if hasattr(layer, 'units'):
                details.append(f"{prefix} Units: {layer.units}")
            if hasattr(layer, 'activation') and layer.activation is not None:
                activation_name = layer.activation.__name__ if hasattr(layer.activation, '__name__') else str(
                    layer.activation)
                details.append(f"{prefix} Activation: {activation_name}")
            if hasattr(layer, 'rate'):
                details.append(f"{prefix} Dropout Rate: {layer.rate}")
            if hasattr(layer, 'return_sequences'):
                details.append(f"{prefix} Return Sequences: {layer.return_sequences}")
            if hasattr(layer, 'return_state'):
                details.append(f"{prefix} Return State: {layer.return_state}")

            layer_info.extend(details)

            if hasattr(layer, 'layers') and len(layer.layers) > 0:
                for sublayer in layer.layers:
                    sublayer_info = process_layer(sublayer, prefix + '  ')
                    layer_info.extend(sublayer_info)

            return layer_info

        # 处理顶层模型
        model_info = process_layer(self.network, '')
        info_lines.extend(model_info)

        return '\n'.join(info_lines)

    def display_information(self, s, level):
        lines = [line for line in s.split('\n') if line.strip()]  # 过滤掉空行
        s_html = '<br>'.join(lines)
        self.network_info.append(f"<div class='{level}' style='margin:0; padding:0;'>{s_html}</div>")
        self.textBrowser.setHtml("".join(self.network_info))

    def Runclick(self):
        start_real_time = time.time()
        if not hasattr(self, 'network'):
            msg_cri("No Network loaded for identification!")
            return
        dummy_input = np.zeros((1, 256, 1), dtype=np.float32)
        dummy_tensor = tf.convert_to_tensor(dummy_input)
        output = self.network(dummy_tensor)  # 前向传播
        output_shape = output.shape[1]
        output_list = [str(i) for i in range(0, output_shape)]
        self.display_information(f'Output Dim: {output_shape}', 'notice')

        self.EXC_dim, ok = QInputDialog.getItem(self, "Which dimension is EXC ?", "Which dimension is EXC\n" +
                                           "list of dims:", output_list, 0, False)
        if not ok:
            return
        self.INH_dim, ok = QInputDialog.getItem(self, "Which dimension is INH ?", "Which dimension is INH\n" +
                                           "list of dims:", output_list, 0, False)
        if not ok:
            return

        LFP = self.signal
        normalized_LFP = normalize_min_max(LFP, -20, 15)
        self.NN_Signal_Window.updatesignal(self.t, normalized_LFP)
        QtWidgets.QApplication.processEvents()

        fs = self.Fs
        dt = 1 / fs
        self.StartTime = int(self.Edit_starttime.text().replace(',', '.'))
        self.EndTime = int(self.Edit_endtime.text().replace(',', '.'))
        target_normalized_LFP = normalized_LFP[self.StartTime * fs: self.EndTime * fs]

        X_list1 = []
        real_time = int(len(target_normalized_LFP) / 256)
        for i in range(real_time):
            target_time = self.Fs * i
            target_signal = target_normalized_LFP[target_time: target_time + self.Fs]
            X_list1.append(target_signal)

        X_list1 = np.stack(X_list1)

        x_mean = target_normalized_LFP.mean()
        x_std = target_normalized_LFP.std()
        y_mean = [[9.97179158, 104.3617891, 33.69248091, 45.07152309]]
        y_std = [[5.02580269, 29.44057363, 10.98829276, 18.2110346]]

        X_test = (X_list1 - x_mean) / x_std
        X_test = np.expand_dims(X_test, axis=2)
        y_pred = self.network.predict(X_test)
        y_pred_expand = y_pred * y_std + y_mean
        self.NN_Signal_Window.updatestate(y_pred_expand, self.EXC_dim, self.INH_dim)
        ...


class Channel_choose_Window(QtWidgets.QDialog, Ui_Channel_choose):
    def __init__(self, parent=None):
        super(Channel_choose_Window, self).__init__(parent)
        self.setupUi(self)
        self.parent = parent
        self.channel_list = self.parent.fieldnames
        self.buttonBox.button(QDialogButtonBox.Ok).clicked.connect(self.Ok_clicked)
        self.updateLayout()
        self.channel_name = None
        self.channel_data = []

    def updateLayout(self):
        grid4 = QGridLayout()
        self.dataset_name = QLabel('Name')
        self.dataset_name.setAlignment(Qt.AlignCenter)
        self.dataset_name.setFont(QFont("Georgia", 10))
        grid4.addWidget(self.dataset_name, 0, 1)

        self.listofedit = []
        for i in np.arange(len(self.channel_list)-1):
            self.listofedit.append([QRadioButton(), QLabel(self.channel_list[i])])
            self.listofedit[i][0].setFixedHeight(30)
            self.listofedit[i][0].setFixedWidth(30)

            self.listofedit[i][1].setAlignment(Qt.AlignCenter)
            self.listofedit[i][1].setFont(QFont("Georgia", 9))
            self.listofedit[i][1].setFixedHeight(30)
            self.listofedit[i][1].setFixedWidth(80)

            grid4.addWidget(self.listofedit[i][0], i + 1, 0)
            grid4.addWidget(self.listofedit[i][1], i + 1, 1)

        self.scrollArea_channel_choose.setFrameShape(QFrame.NoFrame)
        self.scrollArea_channel_choose.setWidgetResizable(True)
        widget = QWidget(self)
        widget.setLayout(grid4)
        self.scrollArea_channel_choose.setWidget(widget)

    def Ok_clicked(self):
        data_freq = int(self.lineEdit_sampling_rate.text())
        data_name = self.lineEdit_nameset.text()
        for p in self.listofedit:
            radiobox = p[0]
            if radiobox.isChecked():
                self.channel_name = p[1].text()
        for row in self.parent.reader:
            if self.channel_name in row:
                self.channel_data.append(row[self.channel_name])
        if data_name is None:
            data_name = self.channel_name
        self.channel_data = np.array(self.channel_data, dtype=float)
        self.parent.loaded_data.append(new_signal_to_plot(Freq=data_freq, Name=data_name, Data=self.channel_data))
        self.accept()


class Signal_preprocess_Window(QtWidgets.QDialog, Ui_Signal_preprocess):
    def __init__(self, parent=None):
        super(Signal_preprocess_Window, self).__init__(parent)
        self.setupUi(self)
        self.parent = parent
        self.color = QColor(0, 255, 255)

        self.gridLayout.setHorizontalSpacing(10)
        self.gridLayout.setVerticalSpacing(5)

        self.pushButton_downsample.clicked.connect(self.Downsample)
        self.pushButton_downsample.setFixedHeight(30)
        self.pushButton_downsample.setFixedWidth(80)

        self.pushButton_filter.clicked.connect(self.Filter)
        self.pushButton_filter.setFixedHeight(30)
        self.pushButton_filter.setFixedWidth(80)

        self.pushButton_cancel.clicked.connect(self.Cancel)
        self.pushButton_apply.clicked.connect(self.Apply)

        self.lineEdit_downsample.setFixedHeight(30)
        self.lineEdit_downsample.setFixedWidth(80)
        self.lineEdit_filter_low.setFixedHeight(30)
        self.lineEdit_filter_low.setFixedWidth(80)
        self.lineEdit_filter_high.setFixedHeight(30)
        self.lineEdit_filter_high.setFixedWidth(80)

        self.index_item = len(self.parent.loaded_data) - 1
        self.Fs = int(self.parent.loaded_data[self.index_item].dictionnaire['Freq'])
        self.signal = copy.copy(self.parent.loaded_data[self.index_item].dictionnaire['Data'])
        self.name = copy.copy(self.parent.loaded_data[self.index_item].dictionnaire['Name'])
        self.t = np.arange(0, len(self.signal)*(1./self.Fs), 1./self.Fs)

        self.signal_window = SignalWindow(self)
        self.verticalLayout_2.addWidget(self.signal_window)
        self.unify_label_sizes()

        self.signal_window.updatelfp()
        self.updateEdit()

    def unify_label_sizes(self):
        label_width = 135
        label_height = 30

        target_labels = [
            "label_filename",
            "label_downsample",
            "label_end",
            "label_start",
            "label_frames",
            "label_filter",
            "label_sampling_rate_2"
        ]

        for name in target_labels:
            label = self.findChild(QLabel, name)
            if label:
                label.setFixedSize(label_width, label_height)
                label.setAlignment(Qt.AlignCenter)

    def Downsample(self):
        if not self.lineEdit_downsample.text():
            msg_cri('The downsample is not valid')
            return
        fs_decimated = int(self.lineEdit_downsample.text())
        downsample_factor = self.Fs // fs_decimated
        remainder = self.Fs % fs_decimated
        if remainder != 0:
            choise = msg_prompt('The downsampling frequency is not an integer multiple of the original frequency, which may cause information loss.\n Do you want to continue?')
            if choise == QMessageBox.No:
                return
            else:
                pass
        self.signal_decimated = decimate(self.signal, downsample_factor)
        self.signal = self.signal_decimated
        self.Fs = fs_decimated
        self.t = np.arange(0, len(self.signal) * (1. / self.Fs), 1. / self.Fs)
        self.signal_window.updatelfp()
        self.updateEdit()

    def Filter(self):
        if not (self.lineEdit_filter_low.text() or self.lineEdit_filter_high.text()):
            msg_cri('The filter is not valid')
            return
        upper_bound = self.lineEdit_filter_high.text()
        lower_bound = self.lineEdit_filter_low.text()
        if not upper_bound:
            highpass_filter = fir_filter_design('highpass', cutoff=float(lower_bound), fs=self.Fs, numtaps=101)
            filtered_signal = apply_filter(self.signal, highpass_filter)
        elif not lower_bound:
            lowpass_filter = fir_filter_design('lowpass', cutoff=float(upper_bound), fs=self.Fs, numtaps=101)
            filtered_signal = apply_filter(self.signal, lowpass_filter)
        else:
            bandpass_filter = fir_filter_design('bandpass', cutoff=[float(lower_bound), float(upper_bound)], fs=self.Fs, numtaps=101)
            filtered_signal = apply_filter(self.signal, bandpass_filter)
        self.signal = filtered_signal
        self.signal_window.updatelfp()
        self.updateEdit()

    def updateEdit(self):
        self.label_filename_edit.setText(self.name)
        self.label_filename_edit.setFont(QFont("Georgia", 11))
        self.label_filename_edit.setFixedHeight(30)
        self.label_filename_edit.setFixedWidth(80)

        self.label_frames_edit.setText(str(len(self.signal)))
        self.label_frames_edit.setAlignment(Qt.AlignCenter)
        self.label_frames_edit.setFont(QFont("Times New Roman", 11))
        self.label_frames_edit.setFixedHeight(30)
        self.label_frames_edit.setFixedWidth(80)

        self.label_sampling_rate_edit.setText(str(self.Fs))
        self.label_sampling_rate_edit.setAlignment(Qt.AlignCenter)
        self.label_sampling_rate_edit.setFont(QFont("Times New Roman", 11))
        self.label_sampling_rate_edit.setFixedHeight(30)
        self.label_sampling_rate_edit.setFixedWidth(80)

        self.label_start_edit.setText('0')
        self.label_start_edit.setAlignment(Qt.AlignCenter)
        self.label_start_edit.setFont(QFont("Times New Roman", 11))
        self.label_start_edit.setFixedHeight(30)
        self.label_start_edit.setFixedWidth(80)

        self.label_end_edit.setText(str(int(len(self.signal)/self.Fs)))
        self.label_end_edit.setAlignment(Qt.AlignCenter)
        self.label_end_edit.setFont(QFont("Times New Roman", 11))
        self.label_end_edit.setFixedHeight(30)
        self.label_end_edit.setFixedWidth(80)

    def Cancel(self):
        self.reject()

    def Apply(self):
        self.parent.preprocessed_signal.append(new_signal_to_plot(Freq=self.Fs, Name=self.name, Data=self.signal))
        self.accept()


class SignalWindow(QGraphicsView):
    def __init__(self, parent=Ui_Signal_preprocess):
        super(SignalWindow, self).__init__(parent)
        self.parent = parent
        self.setStyleSheet("border: none;")
        self.setFrameShape(QFrame.NoFrame)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas)
        self.figure.subplots_adjust(top=0.98, bottom=0.2, left=0.14, right=0.95, hspace=0.1, wspace=0.1)

        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.toolbar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        self.axes = self.figure.add_subplot(1, 1, 1)
        self.axes.set_xlabel("Time (s)", fontsize=13, labelpad=1)
        self.axes.set_ylabel("EEG (a.u.)", fontsize=13, labelpad=5)
        self.axes.tick_params(axis='both', labelsize=13)

    def updatelfp(self):
        self.axes.clear()
        self.axes.set_xlabel("Time (s)", fontsize=13, labelpad=1)
        self.axes.set_ylabel("EEG (a.u.)", fontsize=13, labelpad=5)
        self.axes.plot(self.parent.t, self.parent.signal)
        self.axes.tick_params(axis='both', labelsize=13)
        self.canvas.draw()


class NN_SignalWindow(QGraphicsView):
    def __init__(self, parent):
        super(NN_SignalWindow, self).__init__(parent)
        self.parent = parent
        self.setStyleSheet("border: none;")
        self.setFrameShape(QFrame.NoFrame)

        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.toolbar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        mpl.rcParams['font.family'] = 'serif'
        mpl.rcParams['font.serif'] = ['Times New Roman']
        mpl.rcParams['mathtext.fontset'] = 'stix'
        mpl.rcParams['axes.unicode_minus'] = False

        self.figure.subplots_adjust(top=0.98, bottom=0.08, left=0.12, right=0.98, hspace=0.4, wspace=0.12)
        self.axes = self.figure.add_subplot(4, 1, 1)
        self.axes.set_xlabel("Time (s)", fontsize=13, labelpad=0.5)
        self.axes.set_ylabel("Original (a.u.)", fontsize=13, labelpad=5)
        self.axes.tick_params(axis='both', labelsize=13)

        self.axesEXC = self.figure.add_subplot(4, 1, 2, sharex=self.axes)
        self.axesEXC.set_xlabel("Time (s)", fontsize=13, labelpad=0.5)
        self.axesEXC.set_ylabel("EXC", fontsize=13, labelpad=5)
        self.axesEXC.tick_params(axis='both', labelsize=13)

        self.axesINH = self.figure.add_subplot(4, 1, 3, sharex=self.axes)
        self.axesINH.set_xlabel("Time (s)", fontsize=13, labelpad=0.5)
        self.axesINH.set_ylabel("INH", fontsize=13, labelpad=5)
        self.axesINH.tick_params(axis='both', labelsize=13)

        self.axesEIR = self.figure.add_subplot(4, 1, 4, sharex=self.axes)
        self.axesEIR.set_xlabel("Time (s)", fontsize=13, labelpad=0.5)
        self.axesEIR.set_ylabel("EIR", fontsize=13, labelpad=5)
        self.axesEIR.tick_params(axis='both', labelsize=13)
        self.figure.align_ylabels([self.axesEXC, self.axesINH, self.axes, self.axesEIR])

    def updatesignal(self, t, signal):
        self.axes.clear()
        self.axes.set_xlabel("Time (s)", fontsize=13, labelpad=1)
        self.axes.set_ylabel("EEG (a.u.)", fontsize=13, labelpad=5)
        self.axes.plot(t, signal, color="black", label='Original', alpha=0.4)
        self.axes.tick_params(axis='both', labelsize=13)
        self.axes.legend(loc='upper right', fontsize=10, fancybox=False, framealpha=0.8)
        self.canvas.draw_idle()

    def updatestate(self, y_pred_expand, EXC_dim, INH_dim):
        EXC = y_pred_expand[:, int(EXC_dim)]
        INH = y_pred_expand[:, int(INH_dim)]
        t = np.arange(self.parent.StartTime, self.parent.EndTime, 1)
        EIR = EXC / (EXC + INH)

        self.axes.legend(loc='upper right', fontsize=10, fancybox=False, framealpha=0.8)

        self.axesEXC.plot(t, EXC, label="EXC", color="red")
        # self.axesEXC.legend(loc='upper right', fontsize=10, fancybox=False, framealpha=0.8)
        self.axesEXC.relim()
        self.axesEXC.autoscale_view(scalex=False)

        self.axesINH.plot(t, INH, label="INH", color="blue")
        # self.axesINH.legend(loc='upper right', fontsize=10, fancybox=False, framealpha=0.8)
        self.axesINH.relim()
        self.axesINH.autoscale_view(scalex=False)

        self.axesEIR.plot(t, EIR, label="EIR", color="purple")
        # self.axesEIR.legend(loc='upper right', fontsize=10, fancybox=False, framealpha=0.8)
        self.axesEIR.relim()
        self.axesEIR.autoscale_view(scalex=False)

        self.canvas.draw_idle()


class new_signal_to_plot:
    def __init__(self, Freq=None, Name=None, Data=None):
        self.dictionnaire = dict()
        self.dictionnaire['Freq'] = Freq
        self.dictionnaire['Name'] = Name
        self.dictionnaire['Data'] = Data


def normalize_min_max(data, new_min, new_max):
    data_centered = data - np.mean(data)
    if np.max(data_centered) > 0:  # 避免除以0
        scale_factor = new_max / np.max(data_centered)
        data_scaled = data_centered * scale_factor
    else:
        data_scaled = data_centered  # 如果全部数据 <=0，无需缩放
    return data_scaled


def fir_filter_design(filter_type, cutoff, fs, numtaps=101, pass_zero=True):
    nyquist = fs / 2
    if isinstance(cutoff, (list, tuple)):
        cutoff = [freq / nyquist for freq in cutoff]
    else:
        cutoff = cutoff / nyquist

    if filter_type == 'lowpass':
        h = firwin(numtaps, cutoff, pass_zero='lowpass')
    elif filter_type == 'highpass':
        h = firwin(numtaps, cutoff, pass_zero='highpass')
    elif filter_type == 'bandpass':
        h = firwin(numtaps, cutoff, pass_zero=False)
    elif filter_type == 'bandstop':
        h = firwin(numtaps, cutoff, pass_zero=True)
    return h


def apply_filter(signal, h):
    filtered_signal = filtfilt(h, 1.0, signal)
    return filtered_signal

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    my_pyqt_form = NN_Window()
    my_pyqt_form.show()
    sys.exit(app.exec_())
