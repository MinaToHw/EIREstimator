import os
import sys
import csv
import inspect
import time
import re
import types
import copy
import numpy as np
import matplotlib as mpl
from scipy.signal import decimate, firwin, filtfilt
from PyQt5 import QtWidgets
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from ui.ui_PSO_Window import Ui_PSOWindow
from ui.ui_channel_choose import Ui_Channel_choose
from ui.ui_signal_preprocess import Ui_Signal_preprocess


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


class PSO_Window(QtWidgets.QWidget, Ui_PSOWindow):
    def __init__(self, parent=None):
        super(PSO_Window, self).__init__()
        self.setupUi(self)
        self.updateWidget = None
        self.signal_preprocess_window = None
        self.Button_Run.clicked.connect(self.Runclick)
        self.Button_LoadNMM.clicked.connect(self.LoadModel)
        self.Button_LoadData.clicked.connect(self.LoadData)
        self.Button_SaveRes.clicked.connect(self.SaveResult)
        self.Button_SaveNMMParam.clicked.connect(self.SaveNMMParam)
        self.Button_SaveAlParam.clicked.connect(self.SaveAlgoParam)
        self.color = QColor(255, 255, 255)
        self.PSO_Params = {'inertia': 0.9, 'max_ind_cor_factor': 1.5, 'max_soc_cor_factor': 1.5, 'swarm_size': 40}
        self.Fix_Params = {}
        self.Free_Params = {}
        self.loaded_data = []
        self.preprocessed_signal = []
        self.loaded_data_format = []

        self.PSO_Signal_Window = PSO_SignalWindow(self)
        self.verticalLayout.addWidget(self.PSO_Signal_Window)


    def UpdateLayout(self):
        if self.updateWidget is None:
            self.updateLayout = QtWidgets.QVBoxLayout()
        else:
            self.updateWidget.deleteLater()
            self.updateWidget = None

        grid2 = QGridLayout()
        grid2.setHorizontalSpacing(3)
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

        grid2.addWidget(self.free_param, 0, 0)
        grid2.addWidget(self.Para_Label, 0, 1)
        grid2.addWidget(self.Para_Val, 0, 2)

        try:
            if 'dt' in self.listvariables:
                self.listvariables.remove('dt')

            self.listofedit = []
            for i in np.arange(len(self.listvariables)):
                line_lower_limit = LineEdit('0')
                line_lower_limit.setEnabled(False)
                line_lower_limit.setAlignment(Qt.AlignCenter)

                line_higher_limit = LineEdit(str(getattr(self.monmodel, self.listvariables[i])))
                line_higher_limit.setAlignment(Qt.AlignCenter)

                self.listofedit.append([QCheckBox(),
                                        QLabel(self.listvariables[i]),
                                        line_lower_limit,
                                        line_higher_limit])
                self.listofedit[i][0].setFixedHeight(30)
                self.listofedit[i][0].setFixedWidth(20)

                self.listofedit[i][1].setAlignment(Qt.AlignCenter)
                self.listofedit[i][1].setFont(QFont("Georgia", 9))
                self.listofedit[i][1].setFixedHeight(30)
                self.listofedit[i][1].setFixedWidth(90)

                self.listofedit[i][2].setFixedHeight(30)
                self.listofedit[i][2].setFixedWidth(55)

                self.listofedit[i][3].setFixedHeight(30)
                self.listofedit[i][3].setFixedWidth(55)

                grid2.addWidget(self.listofedit[i][0], i + 1, 0)
                grid2.addWidget(self.listofedit[i][1], i + 1, 1)
                grid2.addWidget(self.listofedit[i][2], i + 1, 2)
                grid2.addWidget(self.listofedit[i][3], i + 1, 3)

                self.listofedit[i][0].toggled.connect(lambda checked, le=line_lower_limit: le.setEnabled(checked))
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

        self.listofedit_PSO = []
        self.PSO_Params_name = list(self.PSO_Params.keys())
        for i in np.arange(len(self.PSO_Params_name)):
            pso_param_value = LineEdit(str(self.PSO_Params[self.PSO_Params_name[i]]))
            self.listofedit_PSO.append([QLabel(self.PSO_Params_name[i]),
                                    pso_param_value])

            self.listofedit_PSO[i][0].setFont(QFont("Georgia", 9))
            self.listofedit_PSO[i][0].setAlignment(Qt.AlignCenter)
            self.listofedit_PSO[i][0].setFixedHeight(30)
            self.listofedit_PSO[i][0].setFixedWidth(150)

            self.listofedit_PSO[i][1].setAlignment(Qt.AlignCenter)
            self.listofedit_PSO[i][1].setFixedHeight(30)
            self.listofedit_PSO[i][1].setFixedWidth(60)

            grid3.addWidget(self.listofedit_PSO[i][0], i + 1, 0)
            grid3.addWidget(self.listofedit_PSO[i][1], i + 1, 1)

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

        (filepath, filename) = os.path.split(fileName)
        sys.path.append(filepath)
        (shortname, extension) = os.path.splitext(filename)
        self.mod = __import__(shortname)
        listclass = sorted(classesinmodule(self.mod))

        # print(listclass)
        item, ok = QInputDialog.getItem(self, "Class Model selection", "Select a Model Class", listclass, 0, False)
        if not ok:
            return
        self.item = str(item)  # item 为 Model
        self.my_class = getattr(self.mod, str(item))
        self.monmodel = self.my_class()
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

        # self.ODE_solver = getattr(self.monmodel, self.mod.get_ODE_solver()[0])  # 减少 self.monmodel 的使用

        try:
            self.listvariables = self.mod.get_Variable_Names()
        except:
            self.listvariables = []
        if self.listvariables == []:
            self.listvariables = sorted(variablesinclass(self.monmodel))

        derivT_source = inspect.getsource(self.monmodel.derivT)
        model_output_formula = None
        for line in derivT_source.splitlines():
            if 'self.LFP =' in line:
                match = re.search(r'self\.LFP\s*=\s*(.*)', line)
                if match:
                    model_output_formula = match.group(1).strip()
                    break

        deriv_source = inspect.getsource(self.monmodel.deriv)
        modified_deriv = re.sub(re.escape(model_output_formula), 'self.exp_data', deriv_source)

        deriv_body = modified_deriv.split(':', 1)[1]  # 提取函数体
        deriv_body_lines = deriv_body.splitlines()  # 按行拆分

        # 去掉原始函数体的错误缩进并重新调整
        adjusted_deriv_body = '\n'.join(['    ' + line.strip() for line in deriv_body_lines if line.strip()])
        new_deriv_code = f"def new_deriv(self):\n{adjusted_deriv_body}"

        # 执行动态代码生成 new_deriv 函数
        exec_namespace = {}
        exec(new_deriv_code, globals(), exec_namespace)
        new_deriv = exec_namespace['new_deriv']
        self.monmodel.new_deriv = types.MethodType(new_deriv, self.monmodel)
        print(new_deriv_code)

        modified_derivT = derivT_source.replace('deriv()', 'new_deriv()')
        derivT_body = modified_derivT.split(':', 1)[1]
        derivT_body_lines = derivT_body.splitlines()
        adjusted_derivT_body = '\n'.join(['    ' + line.strip() for line in derivT_body_lines if line.strip()])
        new_derivT_code = f"def new_derivT(self):\n{adjusted_derivT_body}"
        exec(new_derivT_code, globals(), exec_namespace)
        new_derivT = exec_namespace['new_derivT']
        self.monmodel.new_derivT = types.MethodType(new_derivT, self.monmodel)
        print(new_derivT_code)

        self.UpdateLayout()
        return

    def LoadData(self):
        fileName = QFileDialog.getOpenFileName(self, caption='Load Data', filter="csv (*.csv);;npy (*.npy)")
        if fileName[0] == '':
            return
        if fileName[1] == 'csv (*.csv)':
            with (open(fileName[0], mode='r') as csv_file):
                self.reader = csv.DictReader(csv_file)
                self.fieldnames = self.reader.fieldnames
                self.loaded_data_format = 'csv'

                self.channel_choose_window = Channel_choose_Window(self)
                # self.signal_preprocess_window.exec_()
                if self.channel_choose_window.exec_() == QDialog.Accepted:
                    self.signal_preprocess_window = Signal_preprocess_Window(self)
                    self.signal_preprocess_window.exec_()
                signal_subject = self.preprocessed_signal[0]
                self.signal = signal_subject.dictionnaire['Data']
                self.Fs = signal_subject.dictionnaire['Freq']
                self.t = np.arange(0, len(self.signal) * (1 / signal_subject.dictionnaire['Freq']),
                                   1 / signal_subject.dictionnaire['Freq'])
                self.PSO_Signal_Window.updatesignal(self.t, self.signal)

        if fileName[1] == 'npy (*.npy)':
            try:
                self.signal_data = np.load(fileName[0], allow_pickle=False)
                msg_cri("Please load the correct data format!")
                return
            except ValueError as e:
                if "allow_pickle" in str(e):
                    self.signal_data = np.load(fileName[0], allow_pickle=True).item()
                    self.fieldnames = list(self.signal_data.keys())
                self.loaded_data_format = 'npy'
                self.channel_choose_window = Channel_choose_Window(self)
                if self.channel_choose_window.exec_() == QDialog.Accepted:
                    self.signal_preprocess_window = Signal_preprocess_Window(self)
                    self.signal_preprocess_window.exec_()
                signal_subject = self.preprocessed_signal[0]
                self.signal = signal_subject.dictionnaire['Data']
                self.Fs = signal_subject.dictionnaire['Freq']
                self.t = np.arange(0, len(self.signal) * (1 / signal_subject.dictionnaire['Freq']),
                                   1 / signal_subject.dictionnaire['Freq'])
                self.PSO_Signal_Window.updatesignal(self.t, self.signal)
        self.Edit_frequency.setText(str(self.Fs))


    def Runclick(self):
        start_real_time = time.time()
        if not hasattr(self, 'monmodel'):
            msg_cri("No Model loaded for identification!")
            return

        for q in self.listofedit_PSO:
            param_name = q[0].text()
            self.PSO_Params[param_name] = np.float64(q[1].text().replace(',', '.'))

        self.free_params = 0
        for p in self.listofedit:
            checkbox = p[0]
            if checkbox.isChecked():
                self.free_params = self.free_params + 1
                var_min = np.float64(p[2].text().replace(',', '.'))
                var_max = np.float64(p[3].text().replace(',', '.'))
                self.Free_Params[p[1].text()] = [var_min, var_max]
            else:
                self.Fix_Params[p[1].text()] = np.float64(p[3].text().replace(',', '.'))
        #################################
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
        if self.loaded_data_format == 'csv':
            normalized_LFP = normalize_min_max(LFP, -20, 15)
        if self.loaded_data_format == 'npy':
            normalized_LFP = LFP
        self.PSO_Signal_Window.updatesignal(self.t, normalized_LFP)
        QtWidgets.QApplication.processEvents()

        fs = self.Fs
        dt = 1. / fs
        self.StartTime = int(self.Edit_starttime.text().replace(',', '.'))
        self.EndTime = int(self.Edit_endtime.text().replace(',', '.'))

        self.t_per_simu = np.arange(0, 1.125, dt)
        self.lfp = np.zeros((self.t_per_simu.shape[0], len(self.LFP_Name)), dtype=np.float64)
        self.data_to_save = []

        for t in np.arange(np.float64(self.Edit_starttime.text().replace(',', '.')), np.float64(self.Edit_endtime.text().replace(',', '.')), 1):
            target_time = int(fs * t)
            self.target_experimental_LFP = normalized_LFP[target_time: target_time + int(fs * 1.125)]

            print('Drawing time ' + str(t))
            text = (f"Drawing time {t}\n")
            self.textBrowser.append(text)
            QtWidgets.QApplication.processEvents()

            best_paramset, cost = self.RunPSOJointFitting()
            self.data_to_save.append(best_paramset)
            self.PSO_Signal_Window.updatestate(t, best_paramset, fs, self.target_experimental_LFP)

        end_real_time = time.time()
        elapsed_time = end_real_time - start_real_time
        time_text = f'\n算法运行完成，总耗时: {elapsed_time:.2f} 秒'
        self.textBrowser.append(time_text)
        print(f"PSO算法运行时间: {elapsed_time:.2f} 秒")
        return

    def RunPSOJointFitting(self):
        self.monmodel.init_vector()
        inertia = self.PSO_Params['inertia']
        max_ind_cor_factor = self.PSO_Params['max_ind_cor_factor']
        max_soc_cor_factor = self.PSO_Params['max_soc_cor_factor']
        swarm_size = int(self.PSO_Params['swarm_size'])
        swarm = np.zeros((swarm_size, 4, self.free_params))  # 随机点个数，初始状态（当前位置，速度，历史最优值对应参数，历史最优值），参数具体值

        self.swarm_val_name = []
        self.val_min = np.zeros(self.free_params)
        self.val_max = np.zeros(self.free_params)
        free_param_rank = 0
        for key, value in self.Free_Params.items():
            if isinstance(value, list):
                self.swarm_val_name.append(key)
                self.val_min[free_param_rank] = value[0]
                self.val_max[free_param_rank] = value[1]
                free_param_rank = free_param_rank + 1

        for iter in range(self.free_params):  # 设置自由参数，scatter
            swarm[:, 0, iter] = self.val_min[iter] + np.random.rand(swarm_size) * (self.val_max[iter] - self.val_min[iter])

        swarm[:, 3, 0] = 1e18  # best value so far
        swarm[:, 1, :] = 0  # initial velocity
        # free_num = 0
        vbest = 1e18
        stuck = 0
        count = 0
        delta_threshold = 0.001

        while vbest > delta_threshold:  # 直到符合标准或循环超过一定次数
            count += 1
            for i in range(swarm_size):

                for z in range(self.free_params):
                    swarm[i, 0, z] = swarm[i, 0, z] + swarm[i, 1, z] / 1.3

                free_param_rank = 0
                for key, value in self.Free_Params.items():
                    if isinstance(value, list):
                        setattr(self.monmodel, key, swarm[i, 0, free_param_rank])  # 自由参数在每次循环中都会修正
                        free_param_rank = free_param_rank + 1

                for key1, value1 in self.Fix_Params.items():
                    setattr(self.monmodel, key1, value1)  # 非自由参数

                setattr(self.monmodel, 'H_P', swarm[i, 0, 0])
                setattr(self.monmodel, 'T_P', swarm[i, 0, 1])
                setattr(self.monmodel, 'dt', np.float64(1. / self.Fs))
                vars = np.zeros(self.free_params)
                for z in range(self.free_params):
                    vars[z] = swarm[i, 0, z]

                for k, t in enumerate(self.t_per_simu):  # 索引位置与具体时间
                    self.monmodel.exp_data = self.target_experimental_LFP[k]
                    self.monmodel.new_derivT()
                    self.lfp[k] = self.monmodel.LFP  # 将 monmodel 中 s 变量指代的lfp名称（LFP）赋予 lfp 变量

                cost = self.computeCost()
                if cost < swarm[i, 3, 0]:  # if a new position is better
                    for z in range(self.free_params):
                        swarm[i, 2, z] = vars[z]  # update 最优参数
                    swarm[i, 3, 0] = cost  # and best value

            prev_vbest = vbest
            gbest = np.argmin(swarm[:, 3, 0])
            vbest = swarm[gbest, 3, 0]

            print('Parameter reconstruction for count ' + str(count))
            print('The best particle is ' + str(gbest))
            print('The best parameter set is ' + str(swarm[gbest, 2, :]))
            print('whose cost is ' + str(vbest) + '\n')

            if prev_vbest - vbest < vbest / 100:
                stuck += 1
            else:
                stuck = 0

            if stuck > 25:
                text = (f"Parameter reconstruction for count {count}\n"
                        f"The best particle is {gbest}\n"
                        f"The best parameter set is {swarm[gbest, 2, :]}\n"
                        f"whose cost is {vbest}\n")
                self.textBrowser.append(text)
                QtWidgets.QApplication.processEvents()
                break

            temp_inertia = inertia - (inertia - 0.4) / 100 * (count - 1)

            for i in range(swarm_size):
                for z in range(self.free_params):
                    swarm[i, 1, z] = (temp_inertia * swarm[i, 1, z] + max_ind_cor_factor * np.random.rand() *
                                      (swarm[i, 2, z] - swarm[i, 0, z]) + max_soc_cor_factor * np.random.rand() * (swarm[gbest, 2, z] - swarm[i, 0, z]))
        temp_best_paramset = swarm[gbest, 2, :]
        temp_cost = vbest

        return temp_best_paramset, temp_cost

    def computeCost(self):
        model_output = self.lfp[32:]
        exp_output = self.target_experimental_LFP[32:]
        mean_Model = np.mean(model_output)
        mean_LFP = np.mean(exp_output)

        physiological = 1
        for x in range(len(self.swarm_val_name)):
            newVal = getattr(self.monmodel, self.swarm_val_name[x])
            if newVal < self.val_min[x] or newVal > self.val_max[x]:
                physiological = 0
                break
        if physiological == 0:
            cost = 1e16
        else:
            model_output_rmean = model_output - mean_Model
            exp_output_rmean = (exp_output - mean_LFP).reshape(-1, 1)
            normalized_model_output = normalize_min_max(model_output_rmean, np.min(exp_output_rmean), np.max(exp_output_rmean))

            cost_1 = zncc(normalized_model_output, exp_output_rmean)
            cost_2 = np.sqrt(np.sum((exp_output_rmean - normalized_model_output) ** 2)/len(exp_output_rmean))

            cost = (1 - cost_1) + 2 * cost_2
        return cost

    def SaveResult(self):
        result_data = {}
        result_data["start time"] = self.StartTime
        result_data["end time"] = self.EndTime
        result_data["target signal"] = self.signal
        result_data["fs"] = self.Fs
        result_data["fit result"] = self.PSO_Signal_Window.fit_data_line.get_ydata()
        result_data["param set"] = self.data_to_save
        fileName = QFileDialog.getSaveFileName(self, caption='Save Data', filter="npy (*.npy)")
        if fileName[0] == '':
            return
        file_path = fileName[0]
        if not file_path.endswith('.npy'):
            file_path += '.npy'

        np.save(file_path, result_data)

    def SaveNMMParam(self):
        self.free_params = 0
        for p in self.listofedit:
            checkbox = p[0]
            if checkbox.isChecked():
                self.free_params = self.free_params + 1
                var_min = np.float64(p[2].text().replace(',', '.'))
                var_max = np.float64(p[3].text().replace(',', '.'))
                self.Free_Params[p[1].text()] = [var_min, var_max]
            else:
                self.Fix_Params[p[1].text()] = np.float64(p[3].text().replace(',', '.'))

        fileName = QFileDialog.getSaveFileName(self,'Save Parameters', '', 'Text files (*.txt);;All files (*)')
        if fileName[0] == '':
            return
        file_path = fileName[0]
        if not file_path.endswith('.txt'):
            file_path += '.txt'
        try:
            with open(file_path, 'w') as f:
                f.write("=== Neural Mass Model Parameters ===\n\n")

                f.write("Free Parameters (with bounds):\n")
                f.write("-" * 40 + "\n")
                for param_name, bounds in self.Free_Params.items():
                    f.write(f"{param_name}: [{bounds[0]:.4f}, {bounds[1]:.4f}]\n")

                f.write("\nFixed Parameters:\n")
                f.write("-" * 40 + "\n")
                for param_name, value in self.Fix_Params.items():
                    f.write(f"{param_name}: {value:.4f}\n")

            QMessageBox.information(self, "Success", "Parameters saved successfully")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to save parameters: {str(e)}")

    def SaveAlgoParam(self):
        for q in self.listofedit_PSO:
            param_name = q[0].text()
            self.PSO_Params[param_name] = np.float64(q[1].text().replace(',', '.'))

        fileName = QFileDialog.getSaveFileName(self, 'Save Parameters', '', 'Text files (*.txt);;All files (*)')
        if fileName[0] == '':
            return
        file_path = fileName[0]
        if not file_path.endswith('.txt'):
            file_path += '.txt'
        try:
            with open(file_path, 'w') as f:
                f.write("=== PSO Algorithm Parameters ===\n\n")

                f.write("\nAlgorithm Parameters:\n")
                f.write("-" * 40 + "\n")
                for param_name, value in self.PSO_Params.items():
                    f.write(f"{param_name}: {value:.4f}\n")

            QMessageBox.information(self, "Success", "Parameters saved successfully")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to save parameters: {str(e)}")

class Channel_choose_Window(QtWidgets.QDialog, Ui_Channel_choose):
    def __init__(self, parent=None):
        super(Channel_choose_Window, self).__init__(parent)
        self.setupUi(self)
        self.parent = parent
        self.channel_list = self.parent.fieldnames
        self.data_format = self.parent.loaded_data_format
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
        for i in np.arange(len(self.channel_list)):
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
        if self.data_format == 'csv':
            for row in self.parent.reader:
                if self.channel_name in row:
                    self.channel_data.append(row[self.channel_name])
            if data_name is None:
                data_name = self.channel_name
        if self.data_format == 'npy':
            signal_data = self.parent.signal_data
            if isinstance(signal_data, dict):
                if self.channel_name in signal_data:
                    self.channel_data = signal_data[self.channel_name]
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

        self.label_end_edit.setText(str(int(len(self.signal) / self.Fs)))
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
        self.toolbar = NavigationToolbar(self.canvas, self)
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


class PSO_SignalWindow(QGraphicsView):
    def __init__(self, parent):
        super(PSO_SignalWindow, self).__init__(parent)
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

    def updatestate(self, target_time, state, fs, exp_data):
        EXC_num = self.parent.Selected_EXC_param[1]
        INH_num = self.parent.Selected_INH_param[1]
        t = np.arange(target_time+0.125, target_time+1.125, 1/fs)  # 1s

        free_param_rank = 0
        for key, value in self.parent.Free_Params.items():
            setattr(self.parent.monmodel, key, state[free_param_rank])
            free_param_rank = free_param_rank + 1

        for key1, value1 in self.parent.Fix_Params.items():
            setattr(self.parent.monmodel, key1, value1)

        setattr(self.parent.monmodel, 'H_P', state[0])
        setattr(self.parent.monmodel, 'T_P', state[1])
        setattr(self.parent.monmodel, 'dt', np.float64(1. / fs))

        self.lfp_per_simu = np.zeros((self.parent.t_per_simu.shape[0], len(self.parent.LFP_Name)), dtype=np.float64)
        for k, i in enumerate(self.parent.t_per_simu):  # 索引位置与具体时间
            self.parent.monmodel.exp_data = exp_data[k]
            self.parent.monmodel.new_derivT()
            self.lfp_per_simu[k] = self.parent.monmodel.LFP
        fit_data = self.lfp_per_simu[32:]  # 1s
        exp_output = exp_data[32:]

        model_output_rmean = fit_data - np.mean(fit_data)
        exp_output_rmean = (exp_output - np.mean(exp_output))
        normalized_model_output = normalize_min_max(model_output_rmean, np.min(exp_output_rmean),
                                                    np.max(exp_output_rmean))
        if self.parent.loaded_data_format == 'npy':
            normalized_model_output = normalized_model_output + np.mean(exp_output)

        plot_EXC = np.ones(len(fit_data)) * state[EXC_num]
        plot_INH = np.ones(len(fit_data)) * state[INH_num]
        plot_EIR = plot_EXC / (plot_EXC + plot_INH)

        fit_data_line = None
        for line in self.axes.get_lines():
            if line.get_label() == "fit_data":
                fit_data_line = line
                break
        if fit_data_line is not None:
            existing_x, existing_y = fit_data_line.get_xdata(), fit_data_line.get_ydata()
            new_x = np.concatenate([existing_x, t])
            new_y = np.concatenate([existing_y, normalized_model_output.flatten()])
            fit_data_line.set_data(new_x, new_y)
        else:
            self.axes.plot(t, normalized_model_output, label="fit_data", color="red")
        self.axes.legend(loc='upper right', fontsize=10, fancybox=False, framealpha=0.8)

        if not hasattr(self, 'plot_EXC_line'):
            self.plot_EXC_line, = self.axesEXC.plot(t, plot_EXC, label="EXC", color="red")
        else:
            existing_x, existing_y = self.plot_EXC_line.get_xdata(), self.plot_EXC_line.get_ydata()
            new_x = np.concatenate([existing_x, t])
            new_y = np.concatenate([existing_y, plot_EXC])
            self.plot_EXC_line.set_data(new_x, new_y)
        self.axesEXC.legend(loc='upper right', fontsize=10, fancybox=False, framealpha=0.8)
        self.axesEXC.relim()
        self.axesEXC.autoscale_view(scalex=False)

        if not hasattr(self, 'plot_INH_line'):
            self.plot_INH_line, = self.axesINH.plot(t, plot_INH, label="INH", color="blue")
        else:
            existing_x, existing_y = self.plot_INH_line.get_xdata(), self.plot_INH_line.get_ydata()
            new_x = np.concatenate([existing_x, t])
            new_y = np.concatenate([existing_y, plot_INH])
            self.plot_INH_line.set_data(new_x, new_y)
        self.axesINH.legend(loc='upper right', fontsize=10, fancybox=False, framealpha=0.8)
        self.axesINH.relim()
        self.axesINH.autoscale_view(scalex=False)

        if not hasattr(self, 'plot_EIR_line'):
            self.plot_EIR_line, = self.axesEIR.plot(t, plot_EIR, label="EIR", color="purple")
        else:
            existing_x, existing_y = self.plot_EIR_line.get_xdata(), self.plot_EIR_line.get_ydata()
            new_x = np.concatenate([existing_x, t])
            new_y = np.concatenate([existing_y, plot_EIR])
            self.plot_EIR_line.set_data(new_x, new_y)
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


def zncc(signal1, signal2):
    if len(signal1) != len(signal2):
        raise ValueError("Signals must have the same length.")
    mean1 = np.mean(signal1)
    mean2 = np.mean(signal2)
    numerator = np.sum((signal1 - mean1) * (signal2 - mean2))
    denominator = np.sqrt(np.sum((signal1 - mean1) ** 2) * np.sum((signal2 - mean2) ** 2))
    if denominator == 0:
        return 0
    else:
        return numerator / denominator


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
    my_pyqt_form = PSO_Window()
    my_pyqt_form.show()
    sys.exit(app.exec_())
