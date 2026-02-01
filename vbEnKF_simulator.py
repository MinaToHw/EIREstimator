import os
import sys
import ast
import csv
import copy
import time
import inspect
import numpy as np
import matplotlib as mpl
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from scipy.signal import decimate, firwin, filtfilt
from ui.ui_EnKF_Window import Ui_EnKFWindow
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


class EnKF_Window(QtWidgets.QWidget, Ui_EnKFWindow):
    def __init__(self, parent=None):
        super(EnKF_Window, self).__init__()
        self.setupUi(self)
        self.updateWidget = None
        self.signal_preprocess_window = None
        self.Button_Run.clicked.connect(self.Runclick)
        self.Button_LoadNMM.clicked.connect(self.LoadModel)
        self.Button_LoadData.clicked.connect(self.LoadData)
        self.color = QColor(255, 255, 255)
        self.EnKF_Params = {'Npar': 200, 'R': 50, 'Signal Noi Cov': 0.01, 'Param Noi Cov': 0.002}
        self.Fix_Params = {}
        self.Free_Params = {}
        self.loaded_data = []
        self.preprocessed_signal = []

        self.EnKF_Signal_Window = EnKF_SignalWindow(self)
        self.verticalLayout.addWidget(self.EnKF_Signal_Window)


    def UpdateLayout(self):
        if self.updateWidget is None:
            self.updateLayout = QtWidgets.QVBoxLayout()
        else:
            self.updateWidget.deleteLater()
            self.updateWidget = None
        grid2 = QGridLayout()
        grid2.setHorizontalSpacing(5)
        grid2.setVerticalSpacing(10)

        self.free_param = QLabel('Free')
        self.free_param.setAlignment(Qt.AlignCenter)
        self.free_param.setFont(QFont("Georgia", 10))

        self.Para_Label = QLabel('Name')
        self.Para_Label.setAlignment(Qt.AlignCenter)
        self.Para_Label.setFont(QFont("Georgia", 10))

        self.Para_Val = QLabel('Value')
        self.Para_Val.setAlignment(Qt.AlignCenter)
        self.Para_Val.setFont(QFont("Georgia", 10))

        self.uplimit = QLabel('Upper')
        self.uplimit.setAlignment(Qt.AlignCenter)
        self.uplimit.setFont(QFont("Georgia", 10))

        self.lowlimit = QLabel('Lower')
        self.lowlimit.setAlignment(Qt.AlignCenter)
        self.lowlimit.setFont(QFont("Georgia", 10))

        grid2.addWidget(self.free_param, 0, 0)
        grid2.addWidget(self.Para_Label, 0, 1)
        grid2.addWidget(self.Para_Val, 0, 2)
        grid2.addWidget(self.lowlimit, 0, 3)
        grid2.addWidget(self.uplimit, 0, 4)

        try:
            if 'dt' in self.listvariables:
                self.listvariables.remove('dt')

            self.listofedit = []
            for i in np.arange(len(self.listvariables)):
                # print(i)
                line_lower_limit = LineEdit('0')
                line_lower_limit.setEnabled(False)
                line_lower_limit.setAlignment(Qt.AlignCenter)
                line_higher_limit = LineEdit('0')
                line_higher_limit.setEnabled(False)
                line_higher_limit.setAlignment(Qt.AlignCenter)
                line_value = LineEdit(str(getattr(self.temp_model, self.listvariables[i])))
                line_value.setAlignment(Qt.AlignCenter)

                self.listofedit.append([QCheckBox(),
                                        QLabel(self.listvariables[i]),
                                        line_value,
                                        line_lower_limit,
                                        line_higher_limit])

                self.listofedit[i][0].setFixedHeight(30)
                self.listofedit[i][0].setFixedWidth(20)

                self.listofedit[i][1].setAlignment(Qt.AlignCenter)
                self.listofedit[i][1].setFont(QFont("Georgia", 9))
                self.listofedit[i][1].setFixedHeight(30)
                self.listofedit[i][1].setFixedWidth(120)

                self.listofedit[i][2].setFixedHeight(30)
                self.listofedit[i][2].setFixedWidth(60)

                self.listofedit[i][3].setFixedHeight(30)
                self.listofedit[i][3].setFixedWidth(60)

                self.listofedit[i][4].setFixedHeight(30)
                self.listofedit[i][4].setFixedWidth(60)

                grid2.addWidget(self.listofedit[i][0], i + 1, 0)
                grid2.addWidget(self.listofedit[i][1], i + 1, 1)
                grid2.addWidget(self.listofedit[i][2], i + 1, 2)
                grid2.addWidget(self.listofedit[i][3], i + 1, 3)
                grid2.addWidget(self.listofedit[i][4], i + 1, 4)

                self.listofedit[i][0].toggled.connect(lambda checked, le=line_lower_limit, he=line_higher_limit: self.set_enabled(checked, le, he))
        except:
            pass

        self.scrollArea.setFrameShape(QFrame.NoFrame)
        self.scrollArea.setWidgetResizable(True)
        widget = QWidget(self)
        widget.setLayout(grid2)
        self.scrollArea.setWidget(widget)

        grid3 = QGridLayout()
        grid3.setHorizontalSpacing(5)
        grid3.setVerticalSpacing(10)
        self.Algo_Para_Label = QLabel('Name')
        self.Algo_Para_Label.setAlignment(Qt.AlignCenter)
        self.Algo_Para_Label.setFont(QFont("Georgia", 10))

        self.Algo_Para_Val = QLabel('Value')
        self.Algo_Para_Val.setAlignment(Qt.AlignCenter)
        self.Algo_Para_Val.setFont(QFont("Georgia", 10))

        grid3.addWidget(self.Algo_Para_Label, 0, 0)
        grid3.addWidget(self.Algo_Para_Val, 0, 1)

        self.listofedit_EnKF = []
        self.EnKF_Params_name = list(self.EnKF_Params.keys())
        for i in np.arange(len(self.EnKF_Params_name)):
            enkf_param_value = LineEdit(str(self.EnKF_Params[self.EnKF_Params_name[i]]))
            enkf_param_value.setAlignment(Qt.AlignCenter)
            label = QLabel(self.EnKF_Params_name[i])
            label.setAlignment(Qt.AlignCenter)
            self.listofedit_EnKF.append([label, enkf_param_value])

            self.listofedit_EnKF[i][0].setFont(QFont("Georgia", 9))
            self.listofedit_EnKF[i][0].setFixedHeight(30)
            self.listofedit_EnKF[i][0].setFixedWidth(150)

            self.listofedit_EnKF[i][1].setFixedHeight(30)
            self.listofedit_EnKF[i][1].setFixedWidth(60)

            grid3.addWidget(self.listofedit_EnKF[i][0], i + 1, 0)
            grid3.addWidget(self.listofedit_EnKF[i][1], i + 1, 1)

        self.scrollArea_2.setFrameShape(QFrame.NoFrame)
        self.scrollArea_2.setWidgetResizable(True)
        widget = QWidget(self)
        widget.setLayout(grid3)
        self.scrollArea_2.setWidget(widget)

    def LoadModel(self):
        fileName = QFileDialog.getOpenFileName(self, "Open Data File", "", "data files (*.py)")
        if fileName[0] == '':
            return
        fileName = str(fileName[0])

        (filepath, filename) = os.path.split(fileName)  # 将一个路径拆分成目录路径和文件名两部分
        sys.path.append(filepath)
        (shortname, extension) = os.path.splitext(filename)  # 将文件路径拆分成文件名和扩展名两部分
        self.mod = __import__(shortname)  # JR_test
        listclass = sorted(classesinmodule(self.mod))  # 获取模块中的所有类

        # print(listclass)
        item, ok = QInputDialog.getItem(self, "Class Model selection", "Select a Model Class", listclass, 0, False)
        if not ok:
            return
        self.item = str(item)  # item 为 Model
        self.my_class = getattr(self.mod, str(item))  # 通过 getattr 获取用户选择的类
        self.temp_model = self.my_class()
        self.LFP_Name = self.mod.get_LFP_Name()
        try:
            self.LFP_color = self.mod.get_LFP_color()
        except:
            self.LFP_color = ['b']

        self.Pulses_Names = self.mod.get_Pulse_Names()
        self.PPS_Names = self.mod.get_PPS_Names()
        try:
            self.sig_color = self.mod.get_Colors()
        except:
            self.sig_color = ['b'] * len(self.PPS_Names)

        try:
            self.listvariables = self.mod.get_Variable_Names()
        except:
            self.listvariables = []
        if self.listvariables == []:
            self.listvariables = sorted(variablesinclass(self.temp_model))

        self.UpdateLayout()
        return

    def set_enabled(self, checked, le, he):
        le.setEnabled(checked)
        he.setEnabled(checked)

    def LoadData(self):
        fileName = QFileDialog.getOpenFileName(self, caption='Load Data', filter="csv (*.csv);;mat (*.mat)")
        if fileName[0] == '':
            return
        if fileName[1] == 'csv (*.csv)':
            with (open(fileName[0], mode='r') as csv_file):
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
                self.EnKF_Signal_Window.updatesignal(self.t, self.signal)

    def Runclick(self):
        start_real_time = time.time()
        if not hasattr(self, 'temp_model'):
            msg_cri("No Model loaded for identification!")
            return

        for q in self.listofedit_EnKF:
            param_name = q[0].text()
            self.EnKF_Params[param_name] = np.float64(q[1].text().replace(',', '.'))

        for p in self.listofedit:
            checkbox = p[0]
            if checkbox.isChecked():
                ori_data = np.float64(p[2].text().replace(',', '.'))
                var_min = np.float64(p[3].text().replace(',', '.'))
                var_max = np.float64(p[4].text().replace(',', '.'))
                self.Free_Params[p[1].text()] = [ori_data, var_min, var_max]
            else:
                self.Fix_Params[p[1].text()] = np.float64(p[2].text().replace(',', '.'))

        #######################################################
        free_param_keys = list(self.Free_Params.keys())
        EXC_name, ok = QInputDialog.getItem(self, "Which parameter is EXC ?",
                                             "Which parameter is EXC\n" + "list of free params:", self.Free_Params, 0,
                                             False)
        self.Selected_EXC_param = [EXC_name, free_param_keys.index(EXC_name)]
        if not ok:
            return
        INH_name, ok = QInputDialog.getItem(self, "Which parameter is INH ?",
                                             "Which parameter is INH\n" + "list of free params:", self.Free_Params, 0,
                                             False)
        self.Selected_INH_param = [INH_name, free_param_keys.index(INH_name)]
        if not ok:
            return

        LFP = self.signal
        normalized_LFP = normalize_min_max(LFP, -20, 15)
        self.EnKF_Signal_Window.updatesignal(self.t, normalized_LFP)
        QtWidgets.QApplication.processEvents()

        fs = self.Fs
        dt = 1 / fs
        self.StartTime = int(self.Edit_starttime.text().replace(',', '.'))
        self.EndTime = int(self.Edit_endtime.text().replace(',', '.'))
        target_normalized_LFP = normalized_LFP[self.StartTime * fs: self.EndTime * fs]
        Nt = int((self.EndTime - self.StartTime) / dt)

        Free_Param_Array = [values[0] for values in self.Free_Params.values()]
        Nstate = len(self.Free_Params) + self.temp_model.NbODEs
        xEst = np.zeros(Nstate)
        xEst[self.temp_model.NbODEs:] = np.array(Free_Param_Array)
        PEst = np.eye(Nstate)
        Q = np.diag(np.hstack((self.EnKF_Params['Signal Noi Cov'] * np.ones(self.temp_model.NbODEs), self.EnKF_Params['Param Noi Cov'] * np.ones(len(self.Free_Params)))))
        R = self.EnKF_Params['R']
        Npar = int(self.EnKF_Params['Npar'])

        x_pred = np.zeros((Nt, Nstate))
        eeg_pred = np.zeros(Nt)
        R_save = np.zeros(Nt)
        R_save[0] = R
        P_save = [PEst]

        Model_EnKF = VbEnKFModel(xEst, PEst, Q, R, Npar, dt, self.temp_model, parent=self)
        for t in range(1, Nt):
            z = target_normalized_LFP[t - 1]
            # update model
            Model_EnKF.vbenkf_estimation(z)

            # store data history
            PEst = Model_EnKF.P
            R = (Model_EnKF.b / Model_EnKF.a) * Model_EnKF.R

            x_pred[t, :] = Model_EnKF.X
            eeg_pred[t] = Model_EnKF.zPred[0]
            R_save[t] = R
            P_save.append(PEst)

            err = abs(z - Model_EnKF.zPred[0]) ** 2
            interval = 128
            if np.mod(t + 1, interval) == 0:
                tmp_para = Model_EnKF.X[self.temp_model.NbODEs:]
                ratio = tmp_para[self.Selected_EXC_param[1]] / (tmp_para[self.Selected_INH_param[1]] + tmp_para[self.Selected_EXC_param[1]])
                text = ('#itr.: %d (R = %.4f, err. = %.4f, A = %.4f, B = %.4f, EIR = %.4f)' % ((t + 1) / interval, R, err, tmp_para[self.Selected_EXC_param[1]], tmp_para[self.Selected_INH_param[1]], ratio))
                self.textBrowser.append(text)
                self.EnKF_Signal_Window.updatestate(t, eeg_pred, x_pred, fs)
                QtWidgets.QApplication.processEvents()

        end_real_time = time.time()
        elapsed_time = end_real_time - start_real_time
        time_text = f'\n算法运行完成，总耗时: {elapsed_time:.2f} 秒'
        self.textBrowser.append(time_text)
        print(f"PSO算法运行时间: {elapsed_time:.2f} 秒")

        return



class VbEnKFModel(QtWidgets.QWidget):
    def __init__(self, X, P, Q, R, Npar, dt, orimodel, parent=None):
        super(VbEnKFModel, self).__init__(parent)
        self.parent = parent
        self.X = X
        self.P = P
        self.Q = Q
        self.R = R
        self.model = orimodel
        self.dt = dt
        self.Npar = Npar
        self.NbODEs = self.parent.temp_model.NbODEs

        self.a0 = 1E-3
        self.b0 = 1E-3

        self.a = self.a0
        self.b = self.b0

        self.free_params = []

        for key, value in self.parent.Fix_Params.items():
            setattr(self.model, key, value)

        for key, value in self.parent.Free_Params.items():
            setattr(self.model, key, value[0])
            self.free_params.append([value[1], value[2]])

        setattr(self.model, 'dt', self.dt)

    def state_predict(self):
        X = self.X
        P = self.P
        Q = self.Q
        dt = self.dt
        Npar = self.Npar
        Nstate = len(X)

        X_new = []
        x_sgm = np.random.multivariate_normal(mean=X, cov=P, size=Npar)  # 从当前状态分布生成的粒子集合
        v = np.random.multivariate_normal(mean=np.zeros(len(X)), cov=Q, size=Npar)  # 过程噪声
        for i in range(Npar):
            for idx, (key, value) in enumerate(self.parent.Free_Params.items()):
                setattr(self.model, key, x_sgm[i, self.NbODEs + idx])
            setattr(self.model, 'H_P', x_sgm[i, self.NbODEs + 0])
            setattr(self.model, 'T_P', x_sgm[i, self.NbODEs + 1])

            self.model.y = x_sgm[i, :self.NbODEs]
            self.model.derivT()
            x_array = np.hstack([self.model.y, x_sgm[i, self.NbODEs:]])
            X_new.append(x_array)
        X_sgm = np.array(X_new) + v
        # X_sgm = np.array([self.state_func(x_sgm[i, :6], x_sgm[i, 6:]) + v[i] for i in range(Npar)])  # 经过状态转移函数与过程噪声后的预测粒子集合
        XPred = np.mean(X_sgm, axis=0)  # 计算均值，xt

        # 先验估计
        self.X_sgm_ = x_sgm
        self.X_ = X
        self.P_ = P

        dx = X_sgm.T - XPred[:, np.newaxis]  # 在列上增加维度（Xt-xt）
        PPred = ((dx @ dx.T) / (Npar - 1)) + Q  # P

        self.X = XPred
        self.P = PPred
        self.X_sgm = X_sgm

    def state_update(self):
        z = self.z
        X = self.X
        X_sgm = self.X_sgm
        P = self.P
        R = self.R
        Npar = self.Npar
        dt = self.dt
        a = self.a + 1/2
        b = self.b
        free_param_num = len(self.parent.Free_Params)

        eta = b / a
        D1 = np.zeros((free_param_num, self.NbODEs))
        D2 = np.eye(free_param_num)
        D = np.hstack([D1, D2])
        ub = []
        lb = []
        for i in self.free_params:
            lb.append(i[0])
            ub.append(i[1])
        ub = np.array(ub)
        lb = np.array(lb)
        c = np.zeros(ub.shape)

        model_source = inspect.getsource(self.model.derivT)
        lines = [line.strip() for line in model_source.split('\n')]
        lfp_line = next(line for line in lines if 'self.LFP' in line)
        expr_str = lfp_line.split('=', 1)[1].strip().replace(' ', '')

        terms = parse_expression(expr_str)
        H = np.zeros(free_param_num + self.NbODEs)
        for i in range(terms.shape[1]):
            H[terms[0, i]] = terms[1, i]
        H = np.array([H])

        z_sgm = H @ X_sgm.T  # 观测分布
        zPred = np.mean(z_sgm, axis=1)  # 均值 Yt
        y = z - zPred
        dx = X_sgm.T - X[:, np.newaxis]
        dz = z_sgm - zPred
        Pxz = (dx @ dz.T) / (Npar - 1)
        Pzz = ((dz @ dz.T) / (Npar - 1)) + (eta * R)
        Pzz_inv = np.linalg.inv(Pzz)

        w = np.random.normal(loc=0, scale=eta * R, size=Npar)
        K = Pxz @ Pzz_inv
        X_new = np.mean(X_sgm.T + K @ (z + w - z_sgm), axis=1)  # xt+1
        P_new = P - K @ Pzz @ K.T

        b = b + 1 / 2 * ((z - (H @ X_new)) ** 2) / R + 1 / 2 * np.trace((H @ P_new @ H.T) / R)

        W_inv = np.linalg.inv(P_new)  # Pt+1^(-1)
        L = W_inv @ D.T @ np.linalg.inv(D @ W_inv @ D.T)  # 拉格朗日算子 Pt+1^(-1) * D.T * (D * Pt+1^(-1) * D.T)^(-1)
        value = D @ X_new
        for i in range(len(value)):  # 查找更新后的值超出边界的参数,如果超出，更新为最近边界
            if (value[i] > ub[i]) | (value[i] < lb[i]):
                if value[i] > ub[i]:
                    c[i] = ub[i]
                elif value[i] < lb[i]:
                    c[i] = lb[i]

        X_c = X_new - L @ (D @ X_new - c)  # xt+1 - 拉格朗日算子 * (D * xt+1 - 参数？)
        for i in range(len(value)):  # 使用更新后的参数
            if (value[i] > ub[i]) | (value[i] < lb[i]):
                X_new[i + 6] = X_c[i + 6]

        self.X = X_new
        self.P = P_new
        self.zPred = zPred
        self.S = Pzz
        self.a = a
        self.b = b

    def vbenkf_estimation(self, z):
        self.z = z

        self.state_predict()

        self.state_update()


def parse_expression(expr_str):

    class Visitor(ast.NodeVisitor):
        def __init__(self):
            self.terms = []

        def visit_UnaryOp(self, node):
            sign = 1 if isinstance(node.op, ast.UAdd) else -1
            self.terms.append((None, sign))  # 先记录符号，索引稍后处理
            self.generic_visit(node)

        def visit_BinOp(self, node):
            op_sign = 1 if isinstance(node.op, ast.Add) else -1
            # 先处理左子树（可能包含符号）
            self.visit(node.left)
            # 为右子树添加当前操作符的影响
            if hasattr(self, '_current_sign'):
                self._current_sign *= op_sign
            else:
                self._current_sign = op_sign
            self.visit(node.right)
            if hasattr(self, '_current_sign'):
                del self._current_sign

        def visit_Subscript(self, node):
            # 处理 self.y[index]
            if isinstance(node.value, ast.Attribute) and node.value.attr == 'y':
                index = node.slice.value if isinstance(node.slice, ast.Constant) else node.slice.value.n
                sign = getattr(self, '_current_sign', 1)  # 默认符号为正
                if self.terms and self.terms[-1][0] is None:
                    # 如果前一个条目是未匹配的符号（来自UnaryOp）
                    self.terms[-1] = (index, self.terms[-1][1])
                else:
                    self.terms.append((index, sign))
            self.generic_visit(node)

    tree = ast.parse(expr_str.strip(), mode='eval')
    visitor = Visitor()
    visitor.visit(tree)

    indices, signs = zip(*visitor.terms)
    return np.vstack([indices, signs])


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
        self.axes.set_xlabel("Time (s)", fontsize=13, labelpad=2)
        self.axes.set_ylabel("Signal (a.u.)", fontsize=13, labelpad=5)
        self.axes.tick_params(axis='both', labelsize=13)

    def updatelfp(self):
        self.axes.clear()
        self.axes.set_xlabel("Time (s)", fontsize=13, labelpad=2)
        self.axes.set_ylabel("Signal (a.u.)", fontsize=13, labelpad=5)
        self.axes.plot(self.parent.t, self.parent.signal)
        self.axes.tick_params(axis='both', labelsize=13)
        self.canvas.draw()


class EnKF_SignalWindow(QGraphicsView):
    def __init__(self, parent):
        super(EnKF_SignalWindow, self).__init__(parent)
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

        self.figure.subplots_adjust(top=0.98, bottom=0.06, left=0.12, right=0.98, hspace=0.4, wspace=0.1)
        self.axes = self.figure.add_subplot(4, 1, 1)
        self.axes.set_xlabel("Time (s)", fontsize=13, labelpad=2)
        self.axes.set_ylabel("Signal (a.u.)", fontsize=13, labelpad=5)
        self.axes.tick_params(axis='both', labelsize=13)

        self.axesEXC = self.figure.add_subplot(4, 1, 2, sharex=self.axes)
        self.axesEXC.set_xlabel("Time (s)", fontsize=13, labelpad=2)
        self.axesEXC.set_ylabel("EXC", fontsize=13, labelpad=5)
        self.axesEXC.tick_params(axis='both', labelsize=13)

        self.axesINH = self.figure.add_subplot(4, 1, 3, sharex=self.axes)
        self.axesINH.set_xlabel("Time (s)", fontsize=13, labelpad=2)
        self.axesINH.set_ylabel("INH", fontsize=13, labelpad=5)
        self.axesINH.tick_params(axis='both', labelsize=13)

        self.axesEIR = self.figure.add_subplot(4, 1, 4, sharex=self.axes)
        self.axesEIR.set_xlabel("Time (s)", fontsize=13, labelpad=2)
        self.axesEIR.set_ylabel("EIR", fontsize=13, labelpad=5)
        self.axesEIR.tick_params(axis='both', labelsize=13)
        self.figure.align_ylabels([self.axesEXC, self.axesINH, self.axes, self.axesEIR])

    def updatesignal(self, t, signal):
        self.axes.clear()
        self.axes.set_xlabel("Time (s)", fontsize=13, labelpad=2)
        self.axes.set_ylabel("lfp (a.u.)", fontsize=13, labelpad=5)
        self.axes.plot(t, signal, label="Target Signal", color="black", alpha=0.4)
        self.axes.tick_params(axis='both', labelsize=13)
        self.axes.legend(loc='upper right', fontsize=10, fancybox=False, framealpha=0.8)
        self.canvas.draw_idle()

    def updatestate(self, nt, eeg, state, fs):
        EXC_num = self.parent.Selected_EXC_param[1]
        INH_num = self.parent.Selected_INH_param[1]
        t = np.arange(self.parent.StartTime, self.parent.StartTime + nt/fs, 1/fs)
        fit_data = eeg[1:nt]
        plot_EXC = state[1:nt, self.parent.temp_model.NbODEs + EXC_num]
        plot_INH = state[1:nt, self.parent.temp_model.NbODEs + INH_num]
        plot_EIR = plot_EXC / (plot_EXC + plot_INH)

        fit_data_line = None
        for line in self.axes.get_lines():
            if line.get_label() == "fit_data":
                fit_data_line = line
                break
        if fit_data_line is not None:
            fit_data_line.set_data(t[1:nt], fit_data)
        else:
            self.axes.plot(t[1:nt], fit_data, label="fit_data", color="red")
        self.axes.legend(loc='upper right', fontsize=10, fancybox=False, framealpha=0.8)

        if not hasattr(self, 'plot_EXC_line'):
            self.plot_EXC_line, = self.axesEXC.plot(t[1:nt], plot_EXC, label="EXC", color="red")
        else:
            self.plot_EXC_line.set_data(t[1:nt], plot_EXC)
        self.axesEXC.legend(loc='upper right', fontsize=10, fancybox=False, framealpha=0.8)
        self.axesEXC.relim()
        self.axesEXC.autoscale_view(scalex=False)

        if not hasattr(self, 'plot_INH_line'):
            self.plot_INH_line, = self.axesINH.plot(t[1:nt], plot_INH, label="INH", color="blue")
        else:
            self.plot_INH_line.set_data(t[1:nt], plot_INH)
        self.axesINH.legend(loc='upper right', fontsize=10, fancybox=False, framealpha=0.8)
        self.axesINH.relim()
        self.axesINH.autoscale_view(scalex=False)

        if not hasattr(self, 'plot_EIR_line'):
            self.plot_EIR_line, = self.axesEIR.plot(t[1:nt], plot_EIR, label="EIR", color="purple")
        else:
            self.plot_EIR_line.set_data(t[1:nt], plot_EIR)
        self.axesEIR.legend(loc='upper right', fontsize=10, fancybox=False, framealpha=0.8)
        self.axesEIR.relim()
        self.axesEIR.autoscale_view(scalex=False)

        self.canvas.draw_idle()


class new_signal_to_plot:
    def __init__(self, Freq=None, Name=None, Data=None):
        self.dictionnaire = dict()
        self.dictionnaire['Freq'] = Freq
        self.dictionnaire['Name'] = Name
        self.dictionnaire['Data'] = Data


def classesinmodule(module):
    md = module.__dict__
    return [c for c in md if (inspect.isclass(md[c]))]


def variablesinclass(classdumodel):
    md = dir(classdumodel)
    md = [m for m in md if not m[0] == '_']
    me = []
    for m in md:
        if isinstance(getattr(classdumodel, m), float):
            me.append(m)
    return me


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
    my_pyqt_form = EnKF_Window()
    my_pyqt_form.show()
    sys.exit(app.exec_())

