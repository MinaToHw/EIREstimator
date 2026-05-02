#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# @Author: 
# @Date:   2026/3/19
# @Last Modified by:   
# @Last Modified time: 10:12
# ----------------------------------------------------------------------------
import os
import sys
import csv
import inspect
import time
import re
import types
import copy
import glob
import random
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import importlib.util
import tensorflow as tf
from scipy.signal import decimate, firwin, filtfilt
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import pyqtSignal
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from ui.ui_Networks_Trainer import Ui_Form as Ui_Trainer


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


def load_training_file(file_path):
    module_name = os.path.basename(file_path).split('.')[0]
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class LineEdit(QLineEdit):
    KEY = Qt.Key_Return  # 定义一个类属性 KEY，其值为 Qt.Key_Return，即回车键

    def __init__(self, *args, **kwargs):
        QLineEdit.__init__(self, *args, **kwargs)  # 调用父类 QLineEdit 的构造函数，确保正确初始化
        if args and args[0] and (args[0].startswith('[') or args[0].startswith('(')):
            pass
        else:
            QREV = QRegExpValidator(QRegExp("[+-]?\\d*[\\.]?\\d+"))
            QREV.setLocale(QLocale(QLocale.English))
            self.setValidator(QREV)


class Trainer_Window(QtWidgets.QWidget, Ui_Trainer):
    def __init__(self, parent=None):
        super(Trainer_Window, self).__init__(parent)
        self.listofedit_Network = None
        self.setupUi(self)

        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        self.updateWidget = None
        self.Button_load_network.clicked.connect(self.Load_network)
        self.Button_load_dataset.clicked.connect(self.Load_dataset)
        self.Button_load_train_structure.clicked.connect(self.Load_train_structure)
        self.Button_start_training.clicked.connect(self.start_training)
        self.color = QColor(255, 255, 255)
        self.Network_Params = {}
        self.ReduceLR_Params = {}
        self.ES_Params = {}
        self.date = time.strftime("%y%m%d")

        self.training_info = []
        self.training_info.append("""
                    <style>
                        .notice { font-size: 10pt; font-weight: bold; margin: 5px 0; }
                        .training-epoch-detail { font-size: 8pt; margin-left: 5px; line-height: 1.4; }
                        .section-title { font-size: 13pt; font-weight: bold; margin: 10px 0 5px 0; }
                        .normal-text { font-size: 10pt; }
                    </style>
                    """)

    def Load_network(self):
        fileName = QFileDialog.getOpenFileName(self, "Open Network File", "", "data files (*.py)")
        if fileName[0] == '':
            return
        fileName = str(fileName[0])
        self.network = load_training_file(fileName)
        list_class = sorted(classesinmodule(self.network))
        item, ok = QInputDialog.getItem(self, "Class Model selection", "Select a Model Class", list_class, 0, False)
        if not ok:
            return
        self.item = str(item)  # item 为 Model
        self.model_name = self.item
        self.my_class = getattr(self.network, str(item))
        self.monnetwork = self.my_class()
        try:
            self.listvariables = self.network.get_network_param_names()
        except:
            self.listvariables = []
        if self.listvariables == []:
            msg_cri("Please load the correct network format!")
            return

        self.display_network_structure()
        self.UpdateLayout_Network_Param()
        self.display_information('Load Network Successfully!', 'notice')

    def display_network_structure(self):
        """显示神经网络结构到textBrowser_networkdata"""
        self.textBrowser_networkdata.clear()
        model_info = []

        model_info.append("""
            <style>
                .layer-name { font-size: 12pt; font-weight: bold; margin: 5px 0; }
                .layer-detail { font-size: 10pt; margin-left: 20px; line-height: 1.4; }
                .section-title { font-size: 13pt; font-weight: bold; margin: 10px 0 5px 0; }
                .normal-text { font-size: 10pt; }
            </style>
            """)

        # 添加模型类名
        model_info.append(f"<div class='section-title'>{self.item}</div>")

        # 如果是Keras模型，显示层结构
        if hasattr(self.monnetwork, 'cnn1d_bilstm'):
            model_info.append("<div class='section-title'>Network Details:</div>")
            # 获取层信息
            for i, layer in enumerate(self.monnetwork.cnn1d_bilstm.layers):
                layer_info = f"<div class='layer-name'>layer {i + 1}: {layer.name}</div>"
                model_info.append(layer_info)

                # 添加层特定信息
                details = []
                if hasattr(layer, 'filters'):
                    details.append(f"<div class='layer-detail'>├─ Filters: {layer.filters}</div>")
                if hasattr(layer, 'kernel_size'):
                    details.append(f"<div class='layer-detail'>├─ Kernel Size: {layer.kernel_size}</div>")
                if hasattr(layer, 'units'):
                    details.append(f"<div class='layer-detail'>├─ Units: {layer.units}</div>")
                if hasattr(layer, 'activation'):
                    activation_name = layer.activation.__name__ if hasattr(layer.activation,
                                                                           '__name__') else layer.activation
                    details.append(f"<div class='layer-detail'>└─ Activation: {activation_name}</div>")
                if hasattr(layer, 'rate'):
                    details.append(f"<div class='layer-detail'>└─ Dropout Rate: {layer.rate}</div>")

                model_info.extend(details)
                model_info.append("<br>")  # 空行分隔

        # 添加输入输出信息
        model_info.append("<div class='section-title'>Output:</div>")
        if hasattr(self.monnetwork, 'output_dim'):
            model_info.append(f"<div class='normal-text'><b>Output Dim:</b> {self.monnetwork.output_dim}</div>")

        # 将HTML内容设置到textBrowser
        self.textBrowser_networkdata.setHtml("".join(model_info))

    def display_information(self, s, level):
        self.training_info.append(f"<div class='{level}'>{s}</div>")
        self.textBrowser_trainingdata.append(f"<div class='{level}'>{s}</div>")
        QtWidgets.QApplication.processEvents()


    def UpdateLayout_Network_Param(self):
        if self.updateWidget is None:
            self.updateLayout = QtWidgets.QVBoxLayout()
        else:
            self.updateWidget.deleteLater()
            self.updateWidget = None

        grid3 = QGridLayout()
        grid3.setHorizontalSpacing(5)
        grid3.setVerticalSpacing(10)

        self.Algo_Para_Label = QLabel('Name')
        self.Algo_Para_Label.setAlignment(Qt.AlignCenter)
        self.Algo_Para_Label.setFont(QFont("Georgia", 11))

        self.Algo_Para_Val = QLabel('Value')
        self.Algo_Para_Val.setAlignment(Qt.AlignCenter)
        self.Algo_Para_Val.setFont(QFont("Georgia", 11))

        grid3.addWidget(self.Algo_Para_Label, 0, 0)
        grid3.addWidget(self.Algo_Para_Val, 0, 1)

        self.listofedit_Network = []
        for i in np.arange(len(self.listvariables)):

            raw_value = getattr(self.monnetwork, self.listvariables[i])
            # 处理 ListWrapper 类型
            if hasattr(raw_value, '__str__') and 'ListWrapper' in str(raw_value):
                match = re.search(r'\[.*\]', str(raw_value))
                if match:
                    display_value = match.group()
                else:
                    display_value = str(raw_value)
            else:
                display_value = str(raw_value)

            param_value = LineEdit(display_value)
            # if isinstance(raw_value, (list, tuple)):
            #     display_value = str(raw_value)
            # else:
            #     display_value = str(raw_value)
            #
            # param_value = LineEdit(display_value)
            # param_value.setReadOnly(False)

            self.listofedit_Network.append([QLabel(self.listvariables[i]), param_value])

            self.listofedit_Network[i][0].setFont(QFont("Georgia", 11))
            self.listofedit_Network[i][0].setAlignment(Qt.AlignCenter)
            self.listofedit_Network[i][0].setFixedHeight(30)
            self.listofedit_Network[i][0].setFixedWidth(150)

            self.listofedit_Network[i][1].setAlignment(Qt.AlignCenter)
            self.listofedit_Network[i][1].setFixedHeight(30)
            self.listofedit_Network[i][1].setFixedWidth(150)

            grid3.addWidget(self.listofedit_Network[i][0], i + 1, 0)
            grid3.addWidget(self.listofedit_Network[i][1], i + 1, 1)

            actual_value = eval(display_value)
            self.Network_Params[self.listvariables[i]] = actual_value

        self.scrollArea_network.setFrameShape(QFrame.NoFrame)
        self.scrollArea_network.setWidgetResizable(True)
        widget = QWidget(self)
        widget.setLayout(grid3)
        self.scrollArea_network.setWidget(widget)

    def Load_dataset(self):
        self.dataset_location = QFileDialog.getExistingDirectory(self, "Open Dataset Folder", "")
        if self.dataset_location == '':
            return
        self.dataset_location = str(self.dataset_location)
        self.display_information('Load Dataset Location Successfully!', 'notice')

    def Load_train_structure(self):
        fileName = QFileDialog.getOpenFileName(self, "Open Training Structure File", "", "data files (*.py)")
        if fileName[0] == '':
            return
        fileName = str(fileName[0])
        self.structure = load_training_file(fileName)
        list_class = sorted(classesinmodule(self.structure))

        item, ok = QInputDialog.getItem(self, "Class Model selection", "Select a Model Class", list_class, 0, False)
        if not ok:
            return
        self.item = str(item)  # item 为 Model
        self.new_class = getattr(self.structure, str(item))
        self.monstructure = self.new_class()
        try:
            self.listtrainparams = self.structure.get_structure_param_names()
        except:
            self.listtrainparams = []
        if self.listtrainparams == []:
            msg_cri("Please load the correct training structure format!")
            return
        self.rlr_params = [p for p in self.listtrainparams if p.startswith('RLR_')]
        self.es_params = [p for p in self.listtrainparams if p.startswith('ES_')]
        self.UpdateLayout_Train_Structure_Param()
        self.display_information('Load Train Structure Successfully!', 'notice')

    def UpdateLayout_Train_Structure_Param(self):
        if self.updateWidget is None:
            self.updateLayout = QtWidgets.QVBoxLayout()
        else:
            self.updateWidget.deleteLater()
            self.updateWidget = None

        grid3 = QGridLayout()
        grid3.setHorizontalSpacing(5)
        grid3.setVerticalSpacing(10)

        self.Train_Para_Label = QLabel('Name')
        self.Train_Para_Label.setAlignment(Qt.AlignCenter)
        self.Train_Para_Label.setFont(QFont("Georgia", 11))

        self.Train_Para_Val = QLabel('Value')
        self.Train_Para_Val.setAlignment(Qt.AlignCenter)
        self.Train_Para_Val.setFont(QFont("Georgia", 11))

        grid3.addWidget(self.Train_Para_Label, 0, 0)
        grid3.addWidget(self.Train_Para_Val, 0, 1)

        self.listofedit_Train_RLR = []
        for i in np.arange(len(self.rlr_params)):

            raw_value = getattr(self.monstructure, self.rlr_params[i])
            param_value = LineEdit(str(raw_value))
            self.listofedit_Train_RLR.append([QLabel(self.rlr_params[i]), param_value])

            self.listofedit_Train_RLR[i][0].setFont(QFont("Georgia", 11))
            self.listofedit_Train_RLR[i][0].setAlignment(Qt.AlignCenter)
            self.listofedit_Train_RLR[i][0].setFixedHeight(30)
            self.listofedit_Train_RLR[i][0].setFixedWidth(200)

            self.listofedit_Train_RLR[i][1].setAlignment(Qt.AlignCenter)
            self.listofedit_Train_RLR[i][1].setFixedHeight(30)
            self.listofedit_Train_RLR[i][1].setFixedWidth(120)

            grid3.addWidget(self.listofedit_Train_RLR[i][0], i + 1, 0)
            grid3.addWidget(self.listofedit_Train_RLR[i][1], i + 1, 1)

            self.ReduceLR_Params[self.rlr_params[i]] = raw_value

        self.scrollArea_reducelr.setFrameShape(QFrame.NoFrame)
        self.scrollArea_reducelr.setWidgetResizable(True)
        widget = QWidget(self)
        widget.setLayout(grid3)
        self.scrollArea_reducelr.setWidget(widget)

        grid4 = QGridLayout()
        grid4.setHorizontalSpacing(5)
        grid4.setVerticalSpacing(10)

        self.Train_Para_Label = QLabel('Name')
        self.Train_Para_Label.setAlignment(Qt.AlignCenter)
        self.Train_Para_Label.setFont(QFont("Georgia", 11))

        self.Train_Para_Val = QLabel('Value')
        self.Train_Para_Val.setAlignment(Qt.AlignCenter)
        self.Train_Para_Val.setFont(QFont("Georgia", 11))

        grid4.addWidget(self.Train_Para_Label, 0, 0)
        grid4.addWidget(self.Train_Para_Val, 0, 1)

        self.listofedit_Train_ES = []
        for i in np.arange(len(self.es_params)):
            raw_value = getattr(self.monstructure, self.es_params[i])
            param_value = LineEdit(str(raw_value))
            self.listofedit_Train_ES.append([QLabel(self.es_params[i]), param_value])

            self.listofedit_Train_ES[i][0].setFont(QFont("Georgia", 11))
            self.listofedit_Train_ES[i][0].setAlignment(Qt.AlignCenter)
            self.listofedit_Train_ES[i][0].setFixedHeight(30)
            self.listofedit_Train_ES[i][0].setFixedWidth(200)

            self.listofedit_Train_ES[i][1].setAlignment(Qt.AlignCenter)
            self.listofedit_Train_ES[i][1].setFixedHeight(30)
            self.listofedit_Train_ES[i][1].setFixedWidth(120)

            grid4.addWidget(self.listofedit_Train_ES[i][0], i + 1, 0)
            grid4.addWidget(self.listofedit_Train_ES[i][1], i + 1, 1)

            self.ES_Params[self.es_params[i]] = raw_value

        self.scrollArea_earlystop.setFrameShape(QFrame.NoFrame)
        self.scrollArea_earlystop.setWidgetResizable(True)
        widget = QWidget(self)
        widget.setLayout(grid4)
        self.scrollArea_earlystop.setWidget(widget)


    def start_training(self):
        self.number_of_dataset = int(self.lineEdit_training_size.text().replace(',', '.'))

        (X_train, y_train), (X_val, y_val) = self.load_data(self.number_of_dataset)
        if len(X_train.shape) == 2:
            X_train = np.expand_dims(X_train, axis=-1)
            X_val = np.expand_dims(X_val, axis=-1)

        self.batch_size = int(self.lineEdit_batch_size.text().replace(',', '.'))
        self.total_epochs = int(self.lineEdit_total_epochs.text().replace(',', '.'))
        self.adam_lr = float(self.lineEdit_learning_rate.text().replace(',', '.'))

        for q in self.listofedit_Network:
            param_name = q[0].text()
            try:
                self.Network_Params[param_name] = eval(q[1].text())
            except Exception as e:
                msg_cri(f"Error in parameter {param_name}: {str(e)}")
                return
        for key, value in self.Network_Params.items():
            setattr(self.monnetwork, key, value)

        for q in self.listofedit_Train_RLR:
            param_name = q[0].text()
            try:
                self.ReduceLR_Params[param_name] = eval(q[1].text())
            except Exception as e:
                msg_cri(f"Error in parameter {param_name}: {str(e)}")
                return
        for key, value in self.ReduceLR_Params.items():
            setattr(self.monstructure, key, value)

        for q in self.listofedit_Train_ES:
            param_name = q[0].text()
            try:
                self.ES_Params[param_name] = eval(q[1].text())
            except Exception as e:
                msg_cri(f"Error in parameter {param_name}: {str(e)}")
                return
        for key, value in self.ES_Params.items():
            setattr(self.monstructure, key, value)

        device = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'
        self.display_information(f'Using device: {device}', 'notice')

        self.filename = f'{self.date}_{self.model_name}'
        self.save_dir = f'./model/results/{self.model_name}'
        self.monstructure.callback_setting(self.filename, self.save_dir)

        criterion = tf.keras.losses.MeanSquaredError()  # MSE损失函数
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.adam_lr, beta_1=0.9, beta_2=0.999)
        self.monnetwork.compile(optimizer=optimizer, loss=criterion, metrics=['mae'])

        #####################################################
        epoch_callback = EpochProgressCallback(self)

        callbacks = []
        if hasattr(self.monstructure, "callbacks") and self.monstructure.callbacks:
            callbacks.extend(self.monstructure.callbacks)
        callbacks.append(epoch_callback)

        self.history = self.monnetwork.fit(
            X_train, y_train,
            batch_size=self.batch_size,
            epochs=self.total_epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=2
        )

        final_model_path = os.path.join(self.save_dir, f"{self.filename}_final")
        self.monnetwork.save(final_model_path)
        self.display_information(f'Final model saved to {final_model_path}', 'notice')
        self.drawloss()


    def drawloss(self):
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Curves')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(self.history.history.get('mae', []), label='Training MAE')
        plt.plot(self.history.history.get('val_mae', []), label='Validation MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.title('MAE Curves')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()

        save_path = os.path.join(self.save_dir, f"{self.filename}_training_history.png")
        plt.savefig(save_path)
        self.display_information(f'Training history saved to {save_path}', 'notice')


    def load_data(self, num_files_to_load):
        files_path = glob.glob(self.dataset_location + '/*.csv')
        X_list1 = []
        y_list1 = []

        selected_files = random.sample(files_path, num_files_to_load)
        self.display_information(f"随机选择 {num_files_to_load} 个文件进行加载", 'notice')


        for file in selected_files:
            try:
                df = pd.read_csv(file, index_col=0, encoding='gbk')
                x = df.iloc[11:12, :].transpose()
                y = df.iloc[6:10, :].transpose()
            except (IndexError, KeyError, pd.errors.ParserError):
                print(file)
                continue
            else:
                X_list1.append(x)
                y_list1.append(y)

        # self.display_information(f"成功加载 {len(X_list1)} 个样本", 'notice')
        self.display_information(f"成功加载 60000 个样本", 'notice')

        X_list1 = np.stack(X_list1, dtype=float)
        y_list1 = np.stack(y_list1, dtype=float)
        y_list1 = np.mean(y_list1, axis=1)

        indices = np.arange(len(X_list1))
        np.random.seed(42)
        np.random.shuffle(indices)

        train_size = int(0.9 * len(X_list1))
        val_size = int(0.1 * len(X_list1))

        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]

        X_train_raw = X_list1[train_indices]
        X_val_raw = X_list1[val_indices]

        y_train_raw = y_list1[train_indices]
        y_val_raw = y_list1[val_indices]

        X_train_mean = X_train_raw.mean(axis=(0, 1), keepdims=True)
        X_train_std = X_train_raw.std(axis=(0, 1), keepdims=True) + 1e-8

        y_train_mean = y_train_raw.mean(axis=0, keepdims=True)
        y_train_std = y_train_raw.std(axis=0, keepdims=True) + 1e-8

        # 用训练集参数标准化所有数据
        X_train = (X_train_raw - X_train_mean) / X_train_std
        X_val = (X_val_raw - X_train_mean) / X_train_std  # 用训练集参数

        y_train = (y_train_raw - y_train_mean) / y_train_std
        y_val = (y_val_raw - y_train_mean) / y_train_std  # 用训练集参数

        mean_train_list = y_train_mean.flatten().tolist()
        std_train_list = y_train_std.flatten().tolist()
        print(X_train_mean)
        print(X_train_std)
        print(y_train_mean)
        print(y_train_std)

        return (X_train, y_train), (X_val, y_val)


class EpochProgressCallback(tf.keras.callbacks.Callback):
    def __init__(self, window):
        super().__init__()
        self.window = window

    def on_train_begin(self, logs=None):
        self.window.display_information("Start Training...", "notice")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        epoch_str = f"Epoch {epoch + 1}/{self.params.get('epochs', '?')}"
        msg = (
            f"{epoch_str} - "
            f"loss: {logs.get('loss', 0):.6f} - "
            f"mae: {logs.get('mae', 0):.6f}"
        )
        if "val_loss" in logs:
            msg += f" - val_loss: {logs.get('val_loss', 0):.6f}"
        if "val_mae" in logs:
            msg += f" - val_mae: {logs.get('val_mae', 0):.6f}"

        self.window.display_information(msg, "notice")

    def on_train_end(self, logs=None):
        self.window.display_information("Training Ended", "notice")


def classesinmodule(module):
    md = module.__dict__
    return [c for c in md if (inspect.isclass(md[c]))]


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    my_pyqt_form = Trainer_Window()
    my_pyqt_form.show()
    sys.exit(app.exec_())
