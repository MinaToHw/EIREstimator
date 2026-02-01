from ui.ui_simulator import Ui_Simulator
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import os
import sys
import numpy as np
import pickle
import inspect
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from scipy import signal
import scipy.io
import time
from numba.experimental import jitclass
from numba import boolean, int32, float64, uint8
import struct
import csv
import copy


class LineEdit(QLineEdit):
    KEY = Qt.Key_Return  # 定义一个类属性 KEY，其值为 Qt.Key_Return，即回车键

    def __init__(self, *args, **kwargs):
        QLineEdit.__init__(self, *args, **kwargs)  # 调用父类 QLineEdit 的构造函数，确保正确初始化
        QREV = QRegExpValidator(QRegExp("[+-]?\\d*[\\.]?\\d+"))  # 允许输入正负号、可选的小数点和数字，表示有效的浮点数格式
        QREV.setLocale(QLocale(QLocale.English))  # 设置验证器的区域为 QLocale(QLocale.English)，以确保小数点符号为点（.）
        self.setValidator(QREV)

    def wheelEvent(self, event):
        if QApplication.keyboardModifiers() == Qt.ControlModifier:
            delta = 1 if event.angleDelta().y() > 0 else -1
            val = float(self.text())  # 将当前文本框中的内容转换为浮点数
            val += delta
            self.setText(str(val))
            event.accept()


class UI_Simulator(QDialog, Ui_Simulator):
    def __init__(self):
        super(UI_Simulator,self).__init__()
        self.setupUi(self)
        self.setWindowTitle("Simulator")

        self.color = self.palette().color(QPalette.Window)
        self.t = np.arange(1000)
        self.lfp = np.zeros(1000, dtype=np.float64)
        self.mascenelfp = lfpViewer(self)
        self.verticalLayout_Figure.addWidget(self.mascenelfp)

        self.scrollArea.setFrameShape(QFrame.NoFrame)
        self.scrollArea.setWidgetResizable(True)

        self.updateWidget = None
        self.Button_LoadNMM.clicked.connect(self.load)
        self.Button_Run.clicked.connect(self.Runclick)
        self.Button_SaveParam.clicked.connect(self.saveNMM_Para)
        self.Button_LoadParam.clicked.connect(self.loadNMM_Para)
        self.Button_SaveRes.clicked.connect(self.SaveRes)
        # self.Button_LoadRes.clicked.connect(self.LoadRes)
        # self.Button_SaveRes.setEnabled(True)
        # self.Button_LoadRes.setEnabled(False)
        self.Button_Stop.clicked.connect(self.Stopclick)

        self.stop = False


    def UpdateLayout(self):
        if self.updateWidget is None:
            self.updateLayout = QtWidgets.QVBoxLayout()
        else:
            self.updateWidget.deleteLater()
            self.updateWidget = None

        grid2 = QGridLayout()
        grid2.setColumnStretch(0, 1)
        grid2.setColumnStretch(1, 3)

        self.Para_Label = QLabel('Name')
        self.Para_Label.setAlignment(Qt.AlignCenter)
        self.Para_Label.setFont(QFont("Georgia", 10))

        self.Para_Val = QLabel('Value')
        self.Para_Val.setAlignment(Qt.AlignCenter)
        self.Para_Val.setFont(QFont("Georgia", 10))

        grid2.addWidget(self.Para_Label, 0, 0)
        grid2.addWidget(self.Para_Val, 0, 1)
        try:
            # self.listvariables=sorted(variablesinclass(self.monmodel))
            if 'dt' in self.listvariables:
                self.listvariables.remove('dt')
            self.listofedit=[]
            for i in np.arange(len(self.listvariables)):
                self.listofedit.append([QLabel(self.listvariables[i]),
                                         #QLineEdit(str(self.monmodel.__dict__[self.listvariables[i]])),
                                         LineEdit(str(getattr(self.monmodel, self.listvariables[i]))),
                                         ])
                self.listofedit[i][0].setAlignment(Qt.AlignCenter)
                self.listofedit[i][0].setFont(QFont("Georgia", 9))
                self.listofedit[i][0].setMaximumHeight(30)
                self.listofedit[i][0].setFixedWidth(90)

                self.listofedit[i][1].setAlignment(Qt.AlignCenter)
                self.listofedit[i][1].setFixedHeight(25)
                self.listofedit[i][1].setFixedWidth(90)

                grid2.addWidget(self.listofedit[i][0], i+1, 0)
                grid2.addWidget(self.listofedit[i][1], i+1, 1)

        except:
            pass

        group_box = QtWidgets.QGroupBox("Parameter Setting")
        group_box.setLayout(grid2)

        self.scrollArea.setFrameShape(QFrame.NoFrame)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setWidget(group_box)

        text = (f"Update Layout success!\n")
        self.textBrowser.append(text)
        return

    def updateValue(self):
        for i in range(len(self.listofedit)):
            self.listofedit[i][1].setText(str(getattr(self.monmodel,self.listofedit[i][0].text())))
        return True

    def plus_click(self,label,value,delta):
        try:
            for p in self.listofedit:
                setattr(self.monmodel, p[0].text(), np.float64(p[1].text().replace(',', '.')))  # 对象 属性 属性值
            setattr(self.monmodel, label.text(), np.float64(value.text().replace(',','.')) + np.float64(delta.text().replace(',','.')))
            self.updateValue()
            self.Runclick()
            return True
        except:
            return False

    def moins_click(self,label,value,delta):
        try:
            for p in self.listofedit:
                setattr(self.monmodel, p[0].text(), np.float64(p[1].text().replace(',', '.')))
            setattr(self.monmodel, label.text(), np.float64(value.text().replace(',','.')) - np.float64(delta.text().replace(',','.')))
            self.updateValue()
            self.Runclick()
            return True
        except:
            return False

    def Stopclick(self):
        self.stop=True

    def SaveRes(self,):
        fileName = QFileDialog.getSaveFileName(caption='Save parameters', filter= ".csv (*.csv);;.data (*.data);;.mat (*.mat)")
        if fileName[0] == '':
            return

        Sigs_dict = {}
        Sigs_dict["t"] = self.t

        Fs = int(1/(self.t[1]-self.t[0]))
        lfp = copy.copy(self.lfp)
        pulse = copy.copy(self.Pulses)
        pps = copy.copy(self.PPS)
        tp = self.t

        for i in range(self.lfp.shape[1]):
            try:
                Sigs_dict[self.LFP_Name[i]] = lfp[:, i]
            except:
                Sigs_dict["LFP"+str(i)] = lfp[:, i]

        for i in range(self.Pulses.shape[1]):
            try:
                Sigs_dict[self.Pulses_Names[i]] = pulse[:, i]
            except:
                Sigs_dict["Pulses"+str(i)] = pulse[:, i]

        for i in range(self.PPS.shape[1]):
            try:
                Sigs_dict[self.PPS_Names[i]] = pps[:, i]
            except:
                Sigs_dict["PPS"+str(i)] = pps[:,i]

        if fileName[1] == '.data (*.data)':
            file_pi = open(fileName[0], 'wb')  # 'wb' 以二进制模式写入
            pickle.dump(Sigs_dict, file_pi, -1)  # 将 Sigs_dict 字典对象序列化并写入到打开的文件中，-1 代表使用最高的协议版本进行序列化
            file_pi.close()
        elif fileName[1] == '.mat (*.mat)':
            scipy.io.savemat(fileName[0], mdict=Sigs_dict)
        elif fileName[1] == '.csv (*.csv)':
            f = open(fileName[0], 'w')
            w = csv.writer(f, delimiter='\t', lineterminator='\n')  # 指定分隔符为制表符（\t），行终止符为换行符（\n）
            w.writerow(Sigs_dict.keys())  # 将 Sigs_dict 字典的键写入文件的第一行，作为列标题
            for values in Sigs_dict.values():
                w.writerow(['{:e}'.format(var) for var in values])
            f.close()
        return

    def Runclick(self):
        # try:
        if not hasattr(self, 'monmodel'):
            msg_cri("No Model loaded for identification!")
            return
        self.stop = False

        for p in self.listofedit:
            setattr(self.monmodel, p[0].text(), np.float64(p[1].text().replace(',', '.')))  # 将 monmodel 对象中的值更新为 listofedit 中对应的值

        parametret_val = [np.float64(self.Edit_starttime.text().replace(',', '.')), np.float64(self.Edit_endtime.text().replace(',', '.')), np.float64(self.Edit_frequency.text().replace(',', '.'))]
        self.t = np.arange(parametret_val[0], parametret_val[1] + 1. / parametret_val[2], 1. / parametret_val[2])
        self.lfp = np.zeros((self.t.shape[0], len(self.LFP_Name)), dtype=np.float64)
        self.Pulses = np.zeros((self.t.shape[0], len(self.Pulses_Names)), dtype=np.float64)
        self.PPS = np.zeros((self.t.shape[0], len(self.PPS_Names)), dtype=np.float64)

        self.detail = 0
        try:
            setattr(self.monmodel, 'dt', np.float64(1. / parametret_val[2]))
            self.monmodel.init_vector()
        except:
            pass
        to = time.time()
        for k, t in enumerate(self.t[0: -1]):  # 索引位置与具体时间
            # self.monmodel.derivT()
            self.ODE_solver()  # 计算 k 时刻时，相关的 LFP_Name Pulses_Names 与 PPS_Names
            for idx, s in enumerate(self.LFP_Name):
                self.lfp[k, idx] = getattr(self.monmodel, s)  # 将 monmodel 中 s 变量指代的lfp名称（LFP）赋予 lfp 变量
            for idx, s in enumerate(self.Pulses_Names):
                self.Pulses[k, idx] = getattr(self.monmodel, s)  # 将 monmodel 中 s 变量指代的Pulses名称（FR_P,FR_EXC,FR_INH）分别赋予到相应的行（k）与列（idx）
            for idx, s in enumerate(self.PPS_Names):  # 同上
                self.PPS[k, idx] = getattr(self.monmodel, s)
        print(time.time()-to)
        self.mascenelfp.updatelfp()

        return

    def load(self):
        fileName = QFileDialog.getOpenFileName(self,"Open Data File" , "", "data files (*.py)")
        if fileName[0] == '':
            return
        fileName = str(fileName[0])
        # print(fileName)
        # fileName=fileName.replace('/','\\')

        (filepath, filename) = os.path.split(fileName)  # 将一个路径拆分成目录路径和文件名两部分
        sys.path.append(filepath)
        (shortname, extension) = os.path.splitext(filename)  # 将文件路径拆分成文件名和扩展名两部分
        self.mod = __import__(shortname)  # JR_test
        listclass = sorted(classesinmodule(self.mod))  # 获取模块中的所有类

        # print(listclass)
        item, ok = QInputDialog.getItem(self, "Class Model selection", "Select a Model Class", listclass, 0, False)
        if ok == False:
            return
        self.item = str(item)  # item 为 Model
        self.my_class = getattr(self.mod, str(item))  # 通过 getattr 获取用户选择的类
        self.monmodel = self.my_class()
        self.LFP_Name = self.mod.get_LFP_Name()
        try:
            self.LFP_color = self.mod.get_LFP_color()
        except:
            self.LFP_color =['b']

        self.Pulses_Names = self.mod.get_Pulse_Names()
        self.PPS_Names = self.mod.get_PPS_Names()
        try:
            self.sig_color = self.mod.get_Colors()
        except:
            self.sig_color =['b']*len(self.PPS_Names)

        self.ODE_solver = getattr(self.monmodel, self.mod.get_ODE_solver()[0])  # 减少 self.monmodel 的使用

        try:
            self.listvariables = self.mod.get_Variable_Names()
        except:
            self.listvariables = []
        if self.listvariables == []:
            self.listvariables = sorted(variablesinclass(self.monmodel))
        self.UpdateLayout()
        return

    def saveNMM_Para(self):
        if not hasattr(self,'monmodel'):
            msg_cri("No Model loaded for identification!" )
            return
        listVar = []
        listVal = []
        for p in self.listofedit :
            Var = p[0].text()
            listVar.append(Var)
            Val = np.float64(p[1].text().replace(',','.'))
            listVal.append(Val)
            setattr(self.monmodel, Var, Val)
        print(str(self.mod)  + '<class '+self.item + '>' )
        self.Save_Model(str(self.mod)  + '<class '+self.item + '>' ,listVar,listVal)
        return

    def Save_Model(self,name=None,listVar=None,listVal=None):
        extension = "txt"
        fileName = QFileDialog.getSaveFileName(caption='Save parameters', filter= extension +" (*." + extension +")")
        if (fileName[0] == ''):
            return
        if os.path.splitext(fileName[0])[1] == '':
            fileName= (fileName[0] + '.' + extension, fileName[1])
        if fileName[1] == extension + " (*." + extension + ")":
            f = open(fileName[0], 'w')
            self.write_model(f, name,listVar, listVal)
            f.close()

    def write_model(self,f,name,listVar,listVal):
        f.write("Model_info::\n")
        f.write("Model_Name = " + name + "\n")
        f.write("Nb_NMM = " + str(1) + "\n")
        for idx_n, n in enumerate(listVar):
            f.write(n + "\t")
            f.write(str(listVal[idx_n])+" ")
            f.write("\n")

    def loadNMM_Para(self):
        model, modelname = self.Load_Model()
        if model == None:
            msg_cri("Unable to load model")
            return
        knownkey =[]
        unknownkey =[]
        for key in model:
            try:
                getattr(self.monmodel,key)
                knownkey.append(key)
            except:
                unknownkey.append(key)
        if unknownkey:
            quit_msg = "The current NMM does not match the file\n" \
                       "unknown variables: " + ','.join([str(u) for u in unknownkey]) + "\n" \
                       "Do you want to load only the known parameters?"+"\n"\
                        "known variables: " + ','.join([str(u) for u in knownkey]) + "\n"
            reply =  QMessageBox.question(self, 'Message',
                    quit_msg, QMessageBox.Yes, QMessageBox.No)
            if reply == QMessageBox.No:
                return
        for key, value in model.items():
            try:
                setattr(self.monmodel,key,value)
            except:
                pass
        self.updateValue()

    def Load_Model(self):
        extension = "txt"
        fileName = QFileDialog.getOpenFileName(caption='Load parameters' ,filter= extension +" (*." + extension +")")
        # fileName[0] 指代的是用户选择的文件的完整路径和文件名。
        # fileName[1] 指代的是用户选择的文件的过滤器(filter)，即文件后缀(扩展名)。
        if fileName[0] == '':
            return None,None
        if os.path.splitext(fileName[0])[1] == '':
            fileName= (fileName[0] + '.' + extension , fileName[1])
        if fileName[1] == extension + " (*." + extension + ")":
            f = open(fileName[0], 'r')
            line = f.readline()
            model=None
            modelname=None
            while not("Model_info::" in line or line == ''):
                line = f.readline()
                # 如果当前行不包含字符串 "Model_info::" 并且不是空行，则继续读取文件的下一行
            if "Model_info" in line:  # 如果读取的行中有 Model_info 字符串
                model, modelname, line = self.read_model(f)
            f.close()
            return model, modelname

    def read_model(self, f):
        line = f.readline()  # 在 "Model_info::" 的基础上再向下读一行
        if '=' in line:
            modelname = line.split('=')[-1]  # 提取等号后面的部分作为模型名称，赋值给变量 modelname
            line = f.readline()  # 再向下读一行
        else:
            modelname = ''
        if '=' in line:
            nbmodel = int(line.split('=')[-1])  # 提取模型数量
            line = f.readline()
        else:
            nbmodel = 1

        numero = 0
        # if nbmodel > 1:
        #     numero = NMM_number(nbmodel)

        model = {}

        while not("::" in line or line == ''):  # 行中 既 没有"::" 也 不为空时
            if not (line == '' or line == "\n"):  # 行中 既 不为空 也 不仅包含换行符
                lsplit = line.split("\t")
                name = lsplit[0]
                # 尝试将列表中的第 numero + 1 个元素转换为浮点数，如果转换失败，则保留其作为字符串
                try:
                    val = float(lsplit[numero + 1])
                except:
                    val = lsplit[numero + 1]
                model[name] = val
            line = f.readline()
        return model, modelname, line


class lfpViewer(QGraphicsView):
    def __init__(self, parent=Ui_Simulator):
        super(lfpViewer, self).__init__(parent)
        self.parent = parent
        self.setStyleSheet("border: 0px")
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.setBackgroundBrush(QBrush(self.parent.color))

        mpl.rcParams['font.family'] = 'serif'
        mpl.rcParams['font.serif'] = ['Times New Roman']
        mpl.rcParams['mathtext.fontset'] = 'stix'
        mpl.rcParams['axes.unicode_minus'] = False

        self.figure = Figure(facecolor=[self.parent.color.red()/255,self.parent.color.green()/255,self.parent.color.blue()/255])
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        self.figure.subplots_adjust(top=0.98, bottom=0.08, left=0.15, right=0.95, hspace=0.35, wspace=0.1)
        # self.save_button = QPushButton()
        # self.save_button.setIcon(QIcon(os.path.join('icons','SaveData.png')))
        # self.save_button.setToolTip("Save Figure Data")
        # self.toolbar.addWidget(self.save_button)
        # self.save_button.clicked.connect(self.saveFigData)
        # self.save_button.setEnabled(False)

        # self.load_button = QPushButton()
        # self.load_button.setIcon(QIcon(os.path.join('icons','LoadData.png')))
        # self.load_button.setToolTip("Load Figure Data")
        # self.toolbar.addWidget(self.load_button)
        # self.load_button.clicked.connect(self.loaddatapickle)
        # self.load_button.setEnabled(False)


        self.axesPulses = self.figure.add_subplot(3, 1, 1)
        self.axesPulses.set_xlabel("Time (s)", fontsize=13, labelpad=2)
        self.axesPulses.set_ylabel("Pulses (Hz)", fontsize=13, labelpad=5)
        self.axesPulses.plot(self.parent.t, self.parent.lfp)
        self.axesPulses.tick_params(axis='both', labelsize=13)

        self.axesPPS = self.figure.add_subplot(3, 1, 2, sharex=self.axesPulses)
        self.axesPPS.set_xlabel("Time (s)", fontsize=13, labelpad=2)
        self.axesPPS.set_ylabel("PPS (mV)", fontsize=13, labelpad=5)
        self.axesPPS.plot(self.parent.t, self.parent.lfp)
        self.axesPPS.tick_params(axis='both', labelsize=13)

        self.axes = self.figure.add_subplot(3, 1, 3, sharex=self.axesPulses)
        self.axes.set_xlabel("Time (s)", fontsize=13, labelpad=2)
        self.axes.set_ylabel("LFP (mV)", fontsize=13, labelpad=5)
        self.axes.plot(self.parent.t, self.parent.lfp)
        self.axes.tick_params(axis='both', labelsize=13)

        self.figure.align_ylabels([self.axesPulses, self.axesPPS, self.axes])

        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.toolbar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        #
        # layout.setStretch(0, 0)  # Toolbar should not stretch
        # layout.setStretch(1, 1)  # Canvas should stretch to fill space
        # self.setLayout(layout)

    def updatelfp(self):
        Fs=int(1/(self.parent.t[1]-self.parent.t[0]))
        lfp = copy.copy(self.parent.lfp)
        pulse = copy.copy(self.parent.Pulses)
        pps = copy.copy(self.parent.PPS)

        # cut = int(float(self.parent.lfpplot_Cut_Edit.text()) * Fs)

        tp = self.parent.t[:-2]
        lfp= lfp[:-2,:]
        pulse = pulse[:-2,:]
        pps = pps[:-2,:]

        self.axes.clear()
        self.axes.set_xlabel("Time (s)", fontsize=13, labelpad=2)
        self.axes.set_ylabel("lfp (mV)", fontsize=13, labelpad=5)
        for s in range(len(self.parent.LFP_Name)):
            self.axes.plot(tp,lfp[:,s],self.parent.LFP_color[s])
        self.axes.legend(self.parent.LFP_Name, loc='upper right', fontsize=10, fancybox=False, framealpha=0.8)
        self.axes.tick_params(axis='both', labelsize=13)

        self.axesPulses.clear()
        self.axesPulses.set_xlabel("Time (s)", fontsize=13, labelpad=2)
        self.axesPulses.set_ylabel("Pulses (Hz)", fontsize=13, labelpad=5)
        for s in range(len(self.parent.Pulses_Names)):
            self.axesPulses.plot(tp,pulse[:,s],self.parent.sig_color[s])
        self.axesPulses.legend(self.parent.Pulses_Names, loc='upper right', fontsize=10, fancybox=False, framealpha=0.8)
        self.axesPulses.tick_params(axis='both', labelsize=13)

        self.axesPPS.clear()
        self.axesPPS.set_xlabel("Time (s)", fontsize=13, labelpad=2)
        self.axesPPS.set_ylabel("PPS (mV)", fontsize=13, labelpad=5)
        for s in range(len(self.parent.PPS_Names)):
            self.axesPPS.plot(tp,pps[:,s],self.parent.sig_color[s])
        self.axesPPS.legend(self.parent.PPS_Names, loc='upper right', fontsize=10, fancybox=False, framealpha=0.8)
        self.axesPPS.tick_params(axis='both', labelsize=13)

        self.canvas.draw_idle()

        text = ("Update Signal success!\n")
        self.parent.textBrowser.append(text)


def classesinmodule(module):
    md = module.__dict__
    return [c for c in md if (inspect.isclass(md[c]))]


def variablesinclass(classdumodel):
    md = dir(classdumodel)
    md = [m for m in md if not m[0] == '_']
    me = []
    for m in md:
        if isinstance(getattr(classdumodel,m), float):
            me.append(m)
    return me


def msg_cri(s):
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Critical)
    msg.setText(s)
    msg.setWindowTitle(" ")
    msg.setStandardButtons(QMessageBox.Ok)
    msg.exec_()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    a = UI_Simulator()
    a.show()
    sys.exit(app.exec_())
