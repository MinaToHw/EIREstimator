import csv
import pickle
import sys
import subprocess
import platform
import copy
import os
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from ui.ui_Main_Window_Frame import Ui_MainWindow
from ui.ui_modify import UI_modify


def msg_cri(s):
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Critical)
    msg.setText(s)
    msg.setWindowTitle(" ")
    msg.setStandardButtons(QMessageBox.Ok)
    msg.exec_()


class Main_Window(QtWidgets.QWidget, Ui_MainWindow):
    def __init__(self, parent=None):
        super(Main_Window, self).__init__()
        self.setupUi(self)

        self.photo_viewer = PhotoViewer(self)
        row_start, col_start, row_span, col_span = self.gridLayout.getItemPosition(self.gridLayout.indexOf(self.PhotoViewer))
        self.gridLayout.removeWidget(self.PhotoViewer)
        self.PhotoViewer.deleteLater()
        self.gridLayout.addWidget(self.photo_viewer, row_start, col_start, row_span, col_span)
        self.PhotoViewer = self.photo_viewer

        self.number_group = QtWidgets.QButtonGroup(self.gridLayout_neurons)
        self.number_group.addButton(self.EXC, -2)
        self.number_group.addButton(self.INH, -3)
        self.number_group.addButton(self.Noise, -100)
        self.number_group.addButton(self.Link, -200)

        self.EXC.toggled.connect(self.radio_clicked)
        self.INH.toggled.connect(self.radio_clicked)
        self.Noise.toggled.connect(self.radio_clicked)
        self.Link.toggled.connect(self.radio_clicked)

        self.number_group_ODE_Solvers = QtWidgets.QButtonGroup(self.gridLayout_solvers)
        self.number_group_ODE_Solvers.addButton(self.EULER, -2)
        self.number_group_ODE_Solvers.addButton(self.RK4, -3)

        self.color_Button.clicked.connect(self.change_color)
        self.button_color = None

        self.Add.toggled.connect(self.Addclick)
        self.Rem.toggled.connect(self.Remclick)
        self.Mod.toggled.connect(self.Modclick)
        self.Gen.clicked.connect(self.Genclick)

        saveAction = QAction('Save', self)
        saveAction.setShortcut('Ctrl+S')
        saveAction.setToolTip('Save Neural Mass Diagram')
        saveAction.triggered.connect(self.save)

        loadAction = QAction('Load', self)
        loadAction.setShortcut('Ctrl+O')
        loadAction.setToolTip('Load Neural Mass Diagram')
        loadAction.triggered.connect(self.load)

        exitAction = QAction('Exit', self)
        exitAction.setShortcut('Ctrl+Q')
        exitAction.setToolTip('Exit application')
        exitAction.triggered.connect(self.close)

        newAction = QAction('New', self)
        newAction.setToolTip('new model')
        newAction.triggered.connect(self.newsheet)

        undoAction = QAction('Undo', self)
        undoAction.setShortcut('Ctrl+Z')
        undoAction.setToolTip('Undo')
        undoAction.triggered.connect(self.undo)

        redoAction = QAction('Redo', self)
        redoAction.setShortcut('Shift+Ctrl+Z')
        redoAction.setToolTip('Redo')
        redoAction.triggered.connect(self.redo)

        infoAction = QAction('open graph info', self)
        infoAction.setToolTip('open graph info')
        infoAction.triggered.connect(self.openinfo)

        openviewer = QAction('open viewer GUI', self)
        openviewer.setToolTip('open viewer GUI')
        openviewer.triggered.connect(self.openviewer)

        self.Save_Button.clicked.connect(saveAction.triggered)
        self.Load_Button.clicked.connect(loadAction.triggered)
        self.Exit_Button.clicked.connect(exitAction.triggered)
        self.New_Button.clicked.connect(newAction.triggered)
        self.Undo_Button.clicked.connect(undoAction.triggered)
        self.Redo_Button.clicked.connect(redoAction.triggered)
        self.Info_Button.clicked.connect(infoAction.triggered)
        self.Viewer_Action.clicked.connect(openviewer.triggered)

        self.cellId_e = -1
        self.cellName_e = ''
        self.Graph_Items = []
        self.Graph_Items_Save = []
        self.Graph_Items_Save_redo = []
        self.sizeundomax = 200

    def forundo(self):
        if len(self.Graph_Items_Save) == 0:
            self.Graph_Items_Save_redo = []
            self.Graph_Items_Save.append(copy.deepcopy(self.Graph_Items))
        elif not self.Graph_Items_Save[-1] == self.Graph_Items:
            self.Graph_Items_Save_redo = []
            self.Graph_Items_Save.append(copy.deepcopy(self.Graph_Items))
            if len(self.Graph_Items_Save) >= self.sizeundomax:
                self.Graph_Items_Save.pop(0)

    def undo(self):
        if len(self.Graph_Items_Save) > 1:
            self.Graph_Items_Save_redo.append(self.Graph_Items_Save.pop(-1))
            self.Graph_Items = copy.deepcopy(self.Graph_Items_Save[-1])
        self.PhotoViewer.UpdateScene()

    def redo(self):
        if len(self.Graph_Items_Save_redo) > 0:
            self.Graph_Items = self.Graph_Items_Save_redo.pop(-1)
            self.Graph_Items_Save.append(copy.deepcopy(self.Graph_Items))
        self.PhotoViewer.UpdateScene()

    def printItem(self, item):
        self.textBrowser.setText("Info selected item : \n")
        for key in item.dictionnaire.keys():
            ch = key + ' : ' + str(item.dictionnaire[key])
            self.textBrowser.append(ch)

    def radio_clicked(self):
        if self.number_group.checkedId() in range(-99, -1):
            self.Name_edit.clear()
            self.Hx_edit.clear()
            self.lamda_edit.clear()
            print('cell picked up')
            self.Name_edit.setDisabled(0)
            self.e0_edit.setDisabled(0)
            self.v0_edit.setDisabled(0)
            self.rx_edit.setDisabled(0)
            self.Hx_edit.setDisabled(0)
            self.lamda_edit.setDisabled(0)
            self.Cx_edit.setDisabled(1)
            self.Noi_mean_edit.setDisabled(1)
            self.Noi_std_edit.setDisabled(1)
            if self.number_group.checkedId() == -2:
                self.Name_edit.setText('EXC')
                self.Hx_edit.setText('3.25')
                self.lamda_edit.setText('100')
            elif self.number_group.checkedId() == -3:
                self.Name_edit.setText('INH')
                self.Hx_edit.setText('22')
                self.lamda_edit.setText('50')

        elif self.number_group.checkedId() in [-200]:
            print('link picked up')
            self.Name_edit.clear()
            self.Hx_edit.clear()
            self.lamda_edit.clear()
            self.Name_edit.setDisabled(1)
            self.e0_edit.setDisabled(1)
            self.v0_edit.setDisabled(1)
            self.rx_edit.setDisabled(1)
            self.Hx_edit.setDisabled(1)
            self.lamda_edit.setDisabled(1)
            self.Cx_edit.setDisabled(0)
            self.Noi_mean_edit.setDisabled(1)
            self.Noi_std_edit.setDisabled(1)
        elif self.number_group.checkedId() in [-100]:
            print('Noise picked up')
            self.Name_edit.clear()
            self.Hx_edit.clear()
            self.lamda_edit.clear()
            self.Name_edit.setDisabled(0)
            self.Name_edit.setText('N')
            self.e0_edit.setDisabled(1)
            self.v0_edit.setDisabled(1)
            self.rx_edit.setDisabled(1)
            self.Hx_edit.setDisabled(1)
            self.lamda_edit.setDisabled(1)
            self.Cx_edit.setDisabled(1)
            self.Noi_mean_edit.setDisabled(0)
            self.Noi_std_edit.setDisabled(0)
        return

    def save(self):
        fileName = QFileDialog.getSaveFileName(self, caption='Save Data', filter="txt (*.txt);;nmm (*.nmm)")
        if fileName[0] == '':
            return

        if fileName[1] == "nmm (*.nmm)":
            file_pi = open(fileName[0], 'wb')
            pickle.dump(self.Graph_Items, file_pi, -1)  # what's append when click on load button?
            file_pi.close()
            # print(self.Graph_Items)
        elif fileName[1] == "txt (*.txt)":
            with open(fileName[0], mode='w', newline='') as csv_file:
                fieldnames = new_item_to_plot().dictionnaire.keys()
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()

                for item in self.Graph_Items:
                    string = [None] * len(fieldnames)
                    for i, key in enumerate(fieldnames):
                        if item.dictionnaire[key] is None:
                            string[i] = 'None'
                        elif type(item.dictionnaire[key]) == type(int()):
                            string[i] = str(item.dictionnaire[key])
                        elif type(item.dictionnaire[key]) == type(float()):
                            string[i] = str(item.dictionnaire[key])
                        elif type(item.dictionnaire[key]) == type(''):
                            string[i] = item.dictionnaire[key]
                        elif type(item.dictionnaire[key]) == type(QPointF()):
                            string[i] = '[' + str(item.dictionnaire[key].x()) + ' ' + str(item.dictionnaire[key].y()) + ']'  # s.replace('[','').replace(']','').split(' ')
                        # elif type(item.dictionnaire[key]) == type(tuple()):
                        #     string[i] = '[' + str(item.dictionnaire[key][0]) + ' ' + str(
                        #         item.dictionnaire[key][1]) + ']'  # s.replace('[','').replace(']','').split(' ')
                        elif type(item.dictionnaire[key]) == type(QColor()):
                            string[i] = item.dictionnaire[key].name()
                    print(string)
                    dictionary = dict(zip(fieldnames, string))
                    writer.writerow(dictionary)

    def load(self):
        fileName = QFileDialog.getOpenFileName(self, caption='Load Data', filter="txt (*.txt);;nmm (*.nmm)")
        if fileName[0] == '':
            return
        if fileName[1] == "nmm (*.nmm)":
            filehandler = open(fileName[0], 'rb')
            self.Graph_Items = pickle.load(filehandler)
            filehandler.close()
            # print(self.Graph_Items)
        elif fileName[1] == "txt (*.txt)":
            with open(fileName[0], mode='r') as csv_file:
                reader = csv.DictReader(csv_file)
                fieldnames = reader.fieldnames
                self.Graph_Items = []
                for row in reader:
                    item = new_item_to_plot()
                    for key in fieldnames:
                        if row[key] == 'None':
                            item.dictionnaire[key] = None
                        elif key == 'ID' or key == 'Type' or key == 'cellId_e' or key == 'cellId_r':
                            item.dictionnaire[key] = int(row[key])
                        elif key == 'eq1' or key == 'eq2' or key == 'lfp' or key == 'noise':
                            item.dictionnaire[key] = row[key]
                        elif '[' in row[key]:
                            s = row[key].replace('[', '').replace(']', '').split(' ')
                            item.dictionnaire[key] = QPointF(float(s[0]), float(s[1]))
                        elif '#' in row[key]:
                            item.dictionnaire[key] = QColor(row[key]).name()
                        else:
                            item.dictionnaire[key] = row[key]
                    self.Graph_Items.append(copy.deepcopy(item))

        self.forundo()
        self.PhotoViewer.UpdateScene()
        return

    def newsheet(self):
        self.Graph_Items = []
        reply = QMessageBox.question(self, '', "Are you sure to reset the model?", QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.Graph_Items = []
            self.PhotoViewer.UpdateScene()

    def closeEvent(self, event):
        quit_msg = "Are you sure you want to exit the program?"
        reply = QMessageBox.question(self, 'Message', quit_msg, QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

    def openinfo(self):
        exPopup = InfoTable(self, Graph_Items=self.Graph_Items)
        exPopup.exec_()
        exPopup.deleteLater()

    def openviewer(self):
        if platform.system() == 'Windows':
            subprocess.Popen([sys.executable, os.path.join("Signal_simulator.py")], shell=False)
        elif platform.system() == 'Darwin':
            subprocess.Popen([sys.executable, os.path.join("Signal_simulator.py")], shell=False)
        elif platform.system() == 'Linux':
            subprocess.Popen([sys.executable, os.path.join("Signal_simulator.py")], shell=False)

    def change_color(self):
        color = QtWidgets.QColorDialog.getColor()
        if color.isValid():
            self.button_color = color.name()
            self.color_Button.setStyleSheet(f"background-color: {color.name()}")

    def get_ID_Max(self):
        idmax = -1
        for idx, item in enumerate(self.Graph_Items):
            ID = item.dictionnaire['ID']
            if ID > idmax:
                idmax = ID
        return idmax

    def Addclick(self):
        if self.Rem.isChecked() or self.Mod.isChecked():
            self.Add.setChecked(False)

        elif self.Add.isChecked() == True:
            print('wait for click on the scene')
            if self.number_group.checkedId() in range(-99, -1):
                self.textBrowser.setText("choose a position for the cell")
            elif self.number_group.checkedId() in [-200]:
                self.textBrowser.setText("choose the emitting cell")
            elif self.number_group.checkedId() in [-100]:
                self.textBrowser.setText("choose a position for the Noise block")
            self.textBrowser.show()
        else:
            pass

    def Remclick(self):
        if self.Add.isChecked() or self.Mod.isChecked():
            self.Rem.setChecked(False)
        else:
            self.textBrowser.setText("choose a cell or a link to Remove")

    def Modclick(self):
        if self.Add.isChecked() or self.Rem.isChecked():
            self.Mod.setChecked(False)
        else:
            self.textBrowser.setText("choose a cell or a link to move")

    def Genclick(self):
        # test if all pop and noise are connected at least ones
        pop_noise = []
        for idx, item in enumerate(self.Graph_Items):
            type = item.dictionnaire["Type"]
            if type in range(-100, -1):
                pop_noise.append([item.dictionnaire["ID"], type, False])  # 整合神经集群

        for idx, item in enumerate(self.Graph_Items):
            type = item.dictionnaire["Type"]
            if type in [-200]:  # 遍历所有link
                for p in pop_noise:
                    if item.dictionnaire["cellId_e"] == p[0] or item.dictionnaire["cellId_r"] == p[0]:
                        p[2] = True

        for p in pop_noise:
            if p[2] == False:  # 如果有神经集群没有link与之相连
                idx = findID(self.Graph_Items, ID=p[0])
                qm = QMessageBox
                if p[1] == -100:
                    ans = qm.question(self, '', "Noise " + self.Graph_Items[idx].dictionnaire[
                        "Name"] + " is not connected\n would you like to continue?", qm.Yes | qm.No)
                    if ans == qm.No:
                        return
                else:
                    ans = qm.question(self, '', "Population " + self.Graph_Items[idx].dictionnaire[
                        "Name"] + " is not connected\n would you like to continue?", qm.Yes | qm.No)
                    if ans == qm.No:
                        return

        for idx2, item2 in enumerate(self.Graph_Items):  # give names to FR and OSO
            type2 = item2.dictionnaire["Type"]
            if type2 in range(-99, -1):
                item2.dictionnaire['pulseName'] = 'FR_' + item2.dictionnaire['Name']
                item2.dictionnaire['ppsName'] = 'PSP_' + item2.dictionnaire['Name']

        ################################
        items = []
        POPs_idx = []
        for idx, item in enumerate(self.Graph_Items):  # Seek all pop name to select one as LFP
            if item.dictionnaire['Type'] in range(-99, -1):
                items.append(item.dictionnaire['Name'])
                POPs_idx.append(idx)

        LFPPOP, ok = QInputDialog.getItem(self, "From which pop the LFP is recorded",
                                          "From which pop the LFP is recorded\n" + "list of pops:", items, 0, False)
        Selected_pop = [LFPPOP, POPs_idx[items.index(LFPPOP)]]
        if not ok:
            return

        self.PhotoViewer.update()

        self.textBrowser.setText("Computation of the equation set")

        nbequa = 0

        for idx, item in enumerate(self.Graph_Items):  # nom pour chaque equa diff
            if item.dictionnaire['Type'] in [-200]:
                item.dictionnaire['Cname'] = 'C_' + item.dictionnaire['Name']

        for idx, item in enumerate(self.Graph_Items):  # nom pour chaque equa diff
            if item.dictionnaire['Type'] in range(-99, -1) :
                item.dictionnaire['eq1'] = str(nbequa)
                nbequa += 1
                item.dictionnaire['eq2'] = str(nbequa)
                nbequa += 1

        for idx, item in enumerate(self.Graph_Items):  # nom pour chaque equa diff
            if item.dictionnaire['Type'] in range(-99, -1):
                item.dictionnaire['Hname'] = 'H_' + item.dictionnaire['Name']
                item.dictionnaire['Tname'] = 'T_' + item.dictionnaire['Name']
                item.dictionnaire['e0name'] = 'e0_' + item.dictionnaire['Name']
                item.dictionnaire['v0name'] = 'v0_' + item.dictionnaire['Name']
                item.dictionnaire['rname'] = 'r_' + item.dictionnaire['Name']
                item.dictionnaire['sigmname'] = 'sigm_' + item.dictionnaire['Name']

        for idx, item in enumerate(self.Graph_Items):  # nom pour chaque equa diff
            if item.dictionnaire['Type'] in [-100]:
                item.dictionnaire['meanname'] = 'm_' + item.dictionnaire['Name']
                item.dictionnaire['stdname'] = 's_' + item.dictionnaire['Name']
                item.dictionnaire['noisename'] = 'noise_' + item.dictionnaire['Name']
                item.dictionnaire['noisevarname'] = 'noise_var_' + item.dictionnaire['Name']
                item.dictionnaire['Cname'] = 'C_' + item.dictionnaire['Name']
        ###########################
        for idx, item in enumerate(self.Graph_Items):
            itemid = item.dictionnaire['ID']
            if item.dictionnaire['Type'] in range(-99, -1):  # 找神经集群，找具有以该神经集群为被接收方的连接，收集其发射源(神经集群或者噪声源)的连接
                malist_pop = []
                malist_noise = []
                for idx2, item2 in enumerate(self.Graph_Items):
                    if item2.dictionnaire['Type'] in [-200]:  # 找连接
                        if item2.dictionnaire['cellId_r'] == itemid:  # 找到接收方为该神经集群的连接
                            if self.Graph_Items[findID(self.Graph_Items, item2.dictionnaire['cellId_e'])].dictionnaire['Type'] == -100:  # 如果该连接的发射方为噪声源
                                malist_noise.append(idx2)
                            elif self.Graph_Items[findID(self.Graph_Items, item2.dictionnaire['cellId_e'])].dictionnaire['Type'] in range(-99, -1):
                                malist_pop.append(idx2)
                item.dictionnaire['linkId_r'] = malist_pop  # 该population接收的连接的编号
                item.dictionnaire['linknoise_r'] = malist_noise

        for idx, item in enumerate(self.Graph_Items):  # LFP de chaque cell
            if item.dictionnaire['Type'] in range(-99, -1):
                if item.dictionnaire['ID'] == Selected_pop[1]:
                    chaine_lfp = ''
                    for i, link in enumerate(item.dictionnaire['linkId_r']):
                        if self.Graph_Items[link].dictionnaire['Type'] == -200:
                            itemindex = findID(self.Graph_Items, self.Graph_Items[link].dictionnaire['cellId_e'])
                            if not self.Graph_Items[itemindex].dictionnaire['Type'] == -100:
                                if self.Graph_Items[itemindex].dictionnaire['Type'] in [-2, -6]:
                                    chaine_lfp = chaine_lfp + ' + '
                                elif self.Graph_Items[itemindex].dictionnaire['Type'] in [-3, -4, -5, -7, -8, -9, -10]:
                                    chaine_lfp = chaine_lfp + ' - '
                                chaine_lfp = chaine_lfp + 'self.y[' + self.Graph_Items[self.Graph_Items[link].dictionnaire['cellId_e']].dictionnaire['eq1'] + ']'
                    item.dictionnaire['lfp'] = chaine_lfp

                chaine_PSP = ''
                chaine_PSP = chaine_PSP + 'self.y[' + item.dictionnaire['eq1']+ ']'
                item.dictionnaire['PSP'] = chaine_PSP

                chaine_noise = ''
                for i, link in enumerate(item.dictionnaire['linknoise_r']):  # i=0,link=4
                    if self.Graph_Items[link].dictionnaire['Type'] == -200:
                        chaine_noise = chaine_noise + self.Graph_Items[self.Graph_Items[link].dictionnaire['cellId_e']].dictionnaire['noisevarname']
                        itemindex = self.Graph_Items[link].dictionnaire['cellId_r']
                        result = [(j, item.dictionnaire['cellId_e']) for j, item in enumerate(self.Graph_Items) if item.dictionnaire['cellId_r'] == 1]
                        for index, row in enumerate(result):
                            type_k = self.Graph_Items[row[1]].dictionnaire['Type']
                            if type_k in [-2, -6]:
                                item2 = self.Graph_Items[row[1]]
                                item2.dictionnaire['noise'] = chaine_noise

                chaine_FR = ''
                if item.dictionnaire['ID'] == Selected_pop[1]:
                    chaine_FR = chaine_FR + 'self.' + item.dictionnaire['sigmname'] + '(' + item.dictionnaire['lfp'] + ')'
                else:
                    for i, link in enumerate(item.dictionnaire['linkId_r']):
                        if self.Graph_Items[link].dictionnaire['Type'] == -200:
                            itemindex = findID(self.Graph_Items, self.Graph_Items[link].dictionnaire['cellId_e'])
                            if not self.Graph_Items[itemindex].dictionnaire['Type'] == -100:
                                if self.Graph_Items[itemindex].dictionnaire['Type'] in [-2, -6]:
                                    chaine_FR = chaine_FR + ' + '
                                elif self.Graph_Items[itemindex].dictionnaire['Type'] in [-3, -4, -5, -7, -8, -9, -10]:
                                    chaine_FR = chaine_FR + ' - '
                                chaine_FR = (chaine_FR + 'self.' + item.dictionnaire['sigmname'] + '(self.' + self.Graph_Items[link].dictionnaire['Cname'] +
                                             ' * self.y[' + self.Graph_Items[self.Graph_Items[link].dictionnaire['cellId_e']].dictionnaire['eq1'] + ']' + ')')
                item.dictionnaire['FR'] = chaine_FR


        fileName = QFileDialog.getSaveFileName(self, 'ModelName', QDir.currentPath(), "py Files (*.py)")
        if fileName[0] == '':
            return
        fileName = str(fileName[0])
        f = open(fileName, 'w')
        f.write('import random\n')
        f.write('import numpy as np\n\n')
        # if self.withNumba.isChecked():
        #     f.write('from numba import jitclass\n')
        #     f.write('from numba import int32, float64\n\n')

        f.write('def get_LFP_Name():\n')
        f.write('    return ["LFP"]\n\n')
        f.write('def get_LFP_color():\n')
        f.write('    return ["' + self.Graph_Items[Selected_pop[1]].dictionnaire[
            'color'] + '"]\n\n')  # 是否需要加.name()？
        f.write('def get_Pulse_Names():\n')
        chaine = ""
        for idx2, item2 in enumerate(self.Graph_Items):  # delet assiciated link
            type2 = item2.dictionnaire["Type"]
            if type2 in range(-99, -1):
                chaine += "\'" + item2.dictionnaire["pulseName"] + "\',"

        f.write('    return [' + chaine + ']\n\n')

        f.write('def get_PPS_Names():\n')
        chaine = ""
        for idx2, item2 in enumerate(self.Graph_Items):  # delet assiciated link
            type2 = item2.dictionnaire["Type"]
            if type2 in range(-99, -1):
                chaine += "\'" + item2.dictionnaire["ppsName"] + "\',"
        f.write('    return [' + chaine + ']\n\n')

        f.write('def get_Colors():\n')
        chaine = ""
        for idx2, item2 in enumerate(self.Graph_Items):  # delet assiciated link
            type2 = item2.dictionnaire["Type"]
            if type2 in range(-99, -1):
                chaine += "\'" + item2.dictionnaire["color"] + "\',"  # 同样的问题

        f.write('    return [' + chaine + ']\n\n')
        f.write('def get_ODE_solver():\n')
        f.write('    return ["derivT"]\n\n')
        f.write('def get_ODE_solver_Time():\n')
        f.write('    return ["deriv_Time"]\n\n')
        f.write('def get_Variable_Names():\n')
        f.write('    return [')

        for idx, item in enumerate(self.Graph_Items):  # nom pour chaque equa diff
            if item.dictionnaire['Type'] in range(-99, -1):
                f.write("'" + item.dictionnaire['Hname'] + "',")
                f.write("'" + item.dictionnaire['Tname'] + "',")
                f.write("'" + item.dictionnaire['e0name'] + "',")
                f.write("'" + item.dictionnaire['v0name'] + "',")
                f.write("'" + item.dictionnaire['rname'] + "',")

        for idx, item in enumerate(self.Graph_Items):  # nom pour chaque equa diff
            if item.dictionnaire['Type'] in [-100]:
                f.write("'" + item.dictionnaire['meanname'] + "',")
                f.write("'" + item.dictionnaire['stdname'] + "',")

        for idx, item in enumerate(self.Graph_Items):  # nom pour chaque equa diff
            if item.dictionnaire['Type'] in [-200]:
                if not self.Graph_Items[findID(self.Graph_Items, item.dictionnaire['cellId_e'])].dictionnaire[
                           'Type'] == -100:
                    f.write("'" + item.dictionnaire['Cname'] + "',")
        f.write(']\n\n')

        f.write('class Model:\n')
        f.write('    def __init__(self,):\n')
        # sigmoide param
        for idx, item in enumerate(self.Graph_Items):  # LFP de chaque cell
            if item.dictionnaire['Type'] in range(-99, -1):
                chaine = ''
                chaine = chaine + 'self.' + item.dictionnaire['v0name'] + '=' + str(float(item.dictionnaire['v0']))
                f.write('        ' + chaine + '\n')
                chaine = ''
                chaine = chaine + 'self.' + item.dictionnaire['e0name'] + '=' + str(float(item.dictionnaire['e0']))
                f.write('        ' + chaine + '\n')
                chaine = ''
                chaine = chaine + 'self.' + item.dictionnaire['rname'] + '=' + str(float(item.dictionnaire['rx']))
                f.write('        ' + chaine + '\n')
                chaine = ''
                chaine = chaine + 'self.' + item.dictionnaire['Hname'] + '=' + str(float(item.dictionnaire['Hx']))
                f.write('        ' + chaine + '\n')
                chaine = ''
                chaine = chaine + 'self.' + item.dictionnaire['Tname'] + '=' + str(float(item.dictionnaire['lamda']))
                f.write('        ' + chaine + '\n')
                chaine = ''
                chaine = chaine + 'self.' + item.dictionnaire['pulseName'] + '= 0.'
                f.write('        ' + chaine + '\n')
                chaine = ''
                chaine = chaine + 'self.' + item.dictionnaire['ppsName'] + '= 0.'
                f.write('        ' + chaine + '\n')

        for idx, item in enumerate(self.Graph_Items):  # LFP de chaque cell
            if item.dictionnaire['Type'] in [-100]:
                chaine = ''
                chaine = chaine + 'self.' + item.dictionnaire['meanname'] + '=' + str(
                    float(item.dictionnaire['Noi_mean']))
                f.write('        ' + chaine + '\n')
                chaine = ''
                chaine = chaine + 'self.' + item.dictionnaire['stdname'] + '=' + str(
                    float(item.dictionnaire['Noi_std']))
                f.write('        ' + chaine + '\n')
                chaine = ''
                chaine = chaine + 'self.' + item.dictionnaire['noisevarname'] + '=0.'
                f.write('        ' + chaine + '\n')

        for idx, item in enumerate(self.Graph_Items):  # LFP de chaque cell
            if item.dictionnaire['Type'] in [-200]:
                itemid = findID(self.Graph_Items, item.dictionnaire["cellId_e"])
                if not (self.Graph_Items[itemid].dictionnaire['Type'] == -100):
                    chaine = ''
                    chaine = chaine + 'self.' + item.dictionnaire['Cname'] + '=' + str(float(item.dictionnaire['Cx']))
                    f.write('        ' + chaine + '\n')

        f.write('        self.dt = 1./2048.\n')
        f.write('        self.NbODEs = ' + str(nbequa) + '\n')
        f.write('        self.init_vector( )\n\n')

        f.write('    def init_vector(self):\n')
        f.write('        self.dydt = np.zeros(self.NbODEs)\n')
        f.write('        self.y = np.zeros(self.NbODEs)\n')
        f.write('        self.yt = np.zeros(self.NbODEs)\n')
        f.write('        self.dydx1 = np.zeros(self.NbODEs)\n')
        f.write('        self.dydx2 = np.zeros(self.NbODEs)\n')
        f.write('        self.dydx3 = np.zeros(self.NbODEs)\n')
        f.write('        self.LFP = 0.\n\n')

        for idx, item in enumerate(self.Graph_Items):  # LFP de chaque cell
            if item.dictionnaire['Type'] in [-100]:
                chaine = ''
                chaine = chaine + 'def ' + item.dictionnaire['noisename'] + '(self):'
                f.write('    ' + chaine + '\n')
                chaine = ''
                chaine = chaine + 'return random.gauss(self.' + item.dictionnaire['meanname'] + ',self.' + \
                         item.dictionnaire['stdname'] + ')'
                f.write('        ' + chaine + '\n\n')
        # Sigmoid functions
        for idx, item in enumerate(self.Graph_Items):  # LFP de chaque cell
            if item.dictionnaire['Type'] in range(-99, -1):
                chaine = ''
                chaine = chaine + 'def ' + item.dictionnaire['sigmname'] + '(self,v):'
                f.write('    ' + chaine + '\n')
                chaine = ''
                chaine = chaine + 'return  self.' + item.dictionnaire['e0name'] + '/(1+np.exp(self.' + \
                         item.dictionnaire['rname'] + '*(self.' + item.dictionnaire['v0name'] + '-v)))'
                f.write('        ' + chaine + '\n\n')

        # PSP functions
        chaine = 'def PTW(self,y0,y1,y2,V,v):'
        f.write('    ' + chaine + '\n')
        chaine = 'return (V*v*y0 - 2*v*y2 - v*v*y1)'
        f.write('        ' + chaine + '\n\n')

        # RK4 functions
        chaine = 'def derivT(self):'
        f.write('    ' + chaine + '\n')
        for idx, item in enumerate(self.Graph_Items):  # LFP de chaque cell
            if item.dictionnaire['Type'] in [-100]:
                chaine = ''
                chaine = chaine + 'self.' + item.dictionnaire['noisevarname'] + ' = self.' + item.dictionnaire[
                    'noisename'] + '()'
                f.write('        ' + chaine + '\n')

        if self.number_group_ODE_Solvers.checkedId() == -3:  # rk4
            f.write('        ' + 'self.yt = self.y+0.' + '\n')
            f.write('        ' + 'self.dydx1=self.deriv()' + '\n')
            f.write('        ' + 'self.y = self.yt + self.dydx1 * self.dt / 2' + '\n')
            f.write('        ' + 'self.dydx2=self.deriv()' + '\n')
            f.write('        ' + 'self.y = self.yt + self.dydx2 * self.dt / 2' + '\n')
            f.write('        ' + 'self.dydx3=self.deriv()' + '\n')
            f.write('        ' + 'self.y = self.yt + self.dydx3 * self.dt' + '\n')
            f.write(
                '        ' + 'self.y =self.yt + self.dt/6. *(self.dydx1+2*self.dydx2+2*self.dydx3+self.deriv())' + '\n')
        elif self.number_group_ODE_Solvers.checkedId() == -2:  # euler
            f.write('        ' + 'self.yt = self.y+0.' + '\n')
            f.write('        ' + 'self.dydx1=self.deriv()' + '\n')
            f.write('        ' + 'self.y += (self.dydx1 * self.dt)\n')

        f.write('        ' + 'self.LFP = ' + self.Graph_Items[Selected_pop[1]].dictionnaire['lfp'] + '\n')

        for idx2, item2 in enumerate(self.Graph_Items):  # delet assiciated link
            type2 = item2.dictionnaire["Type"]
            chaine = ''
            if type2 in range(-99, -1):
                f.write("        self." + item2.dictionnaire["ppsName"] + " = " + item2.dictionnaire['PSP'] + '\n')
                f.write("        self." + item2.dictionnaire["pulseName"] + " = " + item2.dictionnaire['FR'] + '\n')

        # write vector solver
        f.write('\n    def deriv_Time(self,N):\n')
        f.write('        lfp = np.zeros(N,)\n')
        f.write('        for k in range(N):\n')
        f.write('            self.derivT()\n')
        f.write('            lfp[k]= self.LFP\n')
        f.write('        return lfp\n\n')

        f.write('    ' + 'def deriv(self):' + '\n')
        # nbequa=0
        for idx, item in enumerate(self.Graph_Items):  # nom pour chaque equa diff
            if item.dictionnaire['Type'] in range(-99, -1):
                if item.dictionnaire['ID'] == Selected_pop[1]:
                    f.write('        self.dydt[' + item.dictionnaire['eq1'] + '] = self.y[' + item.dictionnaire[
                        'eq2'] + ']\n')
                    f.write('        self.dydt[' + item.dictionnaire['eq2'] + '] = self.PTW((')
                    noise_value = 'self.' + item.dictionnaire['noise'] if item.dictionnaire['noise'] is not None else ''
                    f.write(noise_value + '+')
                    f.write(item.dictionnaire['FR'] + '), self.y[' + item.dictionnaire['eq1'] + '],self.y[' + item.dictionnaire['eq2'])
                    f.write('],self.' + item.dictionnaire['Hname'] + ' , self.' + item.dictionnaire['Tname'] + ')\n')
                else:
                    f.write('        self.dydt[' + item.dictionnaire['eq1'] + '] = self.y[' + item.dictionnaire[
                        'eq2'] + ']\n')
                    f.write('        self.dydt[' + item.dictionnaire['eq2'] + '] = self.PTW((')
                    noise_value = 'self.' + item.dictionnaire['noise'] if item.dictionnaire['noise'] is not None else ''
                    f.write(noise_value + '+')
                    result_cname = [item2.dictionnaire['Cname'] for index2, item2 in enumerate(self.Graph_Items) if item2.dictionnaire['cellId_e'] == item.dictionnaire['ID']]
                    f.write('self.' + result_cname[0] + '*' + item.dictionnaire['FR'])
                    f.write('), self.y[' + item.dictionnaire['eq1'] + '],self.y[' + item.dictionnaire['eq2'])
                    f.write('],self.' + item.dictionnaire['Hname'] + ' , self.' + item.dictionnaire['Tname'] + ')\n')

        f.write('\n        ' + 'return self.dydt+0.' + '\n')


class PhotoViewer(QtWidgets.QGraphicsView):
    def __init__(self, parent=None):
        super(PhotoViewer, self).__init__(parent)
        self.setRenderHint(QPainter.Antialiasing)
        self.parent = parent
        self.withouttext = True
        self.linkattente = False
        self.position1 = QPointF()
        self.clickedonitem = -1
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.setBackgroundBrush(QBrush(Qt.white))
        self.scene.update()

    def mouseReleaseEvent(self, event):
        if self.clickedonitem > -1:
            self.parent.forundo()
        self.clickedonitem = -1
        self.positionclickedon = None

    def mouseMoveEvent(self, event):
        if self.clickedonitem > -1:
            if bool(event.buttons() & Qt.LeftButton):  # 如果触发事件的按钮状态为按下，且为鼠标左键
                if self.parent.Add.isChecked() == False and self.parent.Rem.isChecked() == False and \
                        self.parent.Mod.isChecked() == False and self.parent.Gen.isEnabled() == True:  # 并且，添加、移除和调整按钮没有被按下，并且生成按键已被使能。
                    # print('yes')
                    position = QPointF(event.pos())  # event.pos() 返回一个 QPoint 对象，表示鼠标事件发生时鼠标的位置坐标。通过使用 QPointF() 构造函数，可以将 QPoint 对象转换为 QPointF 对象。
                    position = self.mapToScene(int(position.x()), int(position.y()))
                    type = self.parent.Graph_Items[self.clickedonitem].dictionnaire["Type"]
                    if type in range(-99, -1) or type in [-100]:
                        position_c = self.parent.Graph_Items[self.clickedonitem].dictionnaire["Pos"]
                        for idx2, item2 in enumerate(self.parent.Graph_Items):  # delet assiciated link
                            type2 = item2.dictionnaire["Type"]
                            if type2 in [-200]:
                                if item2.dictionnaire["cellId_e"] == \
                                        self.parent.Graph_Items[self.clickedonitem].dictionnaire['ID']:
                                    self.parent.Graph_Items[idx2].dictionnaire['Pos'] = position
                                elif item2.dictionnaire["cellId_r"] == \
                                        self.parent.Graph_Items[self.clickedonitem].dictionnaire['ID']:
                                    self.parent.Graph_Items[idx2].dictionnaire['Pos2'] = position
                        self.parent.Graph_Items[self.clickedonitem].dictionnaire['Pos'] = position
                    if type in [-200]:
                        self.parent.Graph_Items[self.clickedonitem].dictionnaire['Pos_c'] = position
                    self.UpdateScene()
        elif self.clickedonitem == -1 and bool(event.buttons() & Qt.LeftButton):  # 如果在没有点击任何项的情况下按下鼠标左键，根据鼠标的移动来平移场景的位置。通过将之前点击的位置和当前鼠标事件的位置进行计算，确定移动的偏移量，并将场景的矩形区域相应地进行平移。
            # print('right!')
            try:
                p1 = QPointF(self.positionclickedon)
                p2 = QPointF(self.mapToScene(event.pos()))
                r = self.sceneRect()
                r.translate(p1 - p2)
                self.setSceneRect(r)
            except:
                pass

    def mousePressEvent(self, event, rightclickinfp=None):
        try:
            if not bool(event.buttons() & Qt.LeftButton):
                self.linkattente = False
            if self.linkattente == True:
                if self.parent.Rem.isChecked() == True or \
                        self.parent.Mod.isChecked() == True or \
                        not self.parent.number_group.checkedId() in [-200] or \
                        event.modifiers() == Qt.ShiftModifier or \
                        event.modifiers() == Qt.ControlModifier:
                    self.linkattente = False

        except:
            pass

        if self.parent.Add.isChecked() == True or self.linkattente == True:

            if self.parent.number_group.checkedId() in range(-100, -1):
                popNames = getNames(self.parent.Graph_Items)
                if self.parent.Name_edit.text().replace(',', '.') in popNames:
                    while self.parent.Name_edit.text().replace(',', '.') in popNames:
                        text, ok = QInputDialog.getText(self, '', "The  Name '" +
                                                        self.parent.Name_edit.text().replace(',', '.') +
                                                        "' Already exists\n Please, enter another one:")
                        if ok:
                            self.parent.Name_edit.setText(text)
                        if not ok:
                            return

            ID = self.parent.get_ID_Max() + 1

            if self.parent.number_group.checkedId() in range(-99, -1):
                position = QPointF(event.pos())
                position = self.mapToScene(int(position.x()), int(position.y()))
                self.parent.Graph_Items.append(new_item_to_plot(ID=ID,
                                                         Type=self.parent.number_group.checkedId(),
                                                         Name=self.parent.Name_edit.text().replace(',', '.'),
                                                         e0=self.parent.e0_edit.text().replace(',', '.'),
                                                         v0=self.parent.v0_edit.text().replace(',', '.'),
                                                         rx=self.parent.rx_edit.text().replace(',', '.'),
                                                         Hx=self.parent.Hx_edit.text().replace(',', '.'),
                                                         lamda=self.parent.lamda_edit.text().replace(',', '.'),
                                                         color=self.parent.button_color,
                                                         Pos=position))

                self.UpdateScene()
                # print(position)
                self.parent.Add.setChecked(0)

            elif self.parent.number_group.checkedId() in [-100]:  # 如果要添加的是噪声源
                position = QPointF(event.pos())
                position = self.mapToScene(int(position.x()), int(position.y()))
                self.parent.Graph_Items.append(new_item_to_plot(ID=ID,
                                                         Type=self.parent.number_group.checkedId(),
                                                         Name=self.parent.Name_edit.text().replace(',', '.'),
                                                         Noi_mean=self.parent.Noi_mean_edit.text().replace(',', '.'),
                                                         Noi_std=self.parent.Noi_std_edit.text().replace(',', '.'),
                                                         Pos=position,
                                                         color=self.parent.button_color))

                self.UpdateScene()
                self.parent.Add.setChecked(0)

            elif self.parent.number_group.checkedId() in [-200]:
                if self.linkattente == False:
                    self.linkattente = True
                    self.position1 = QPointF(event.pos())
                    self.position1 = self.mapToScene(self.position1.x(), self.position1.y())
                    self.cellId_e = -1
                    self.cellName_e = ''
                    length = np.inf
                    for idx, item in enumerate(self.parent.Graph_Items):
                        # print(item)
                        itemID = item.dictionnaire["ID"]
                        type = item.dictionnaire["Type"]
                        nom = item.dictionnaire["Name"]
                        position_c = item.dictionnaire["Pos"]
                        if type in range(-99, -1) or type in [-100]:
                            length_c = ((self.position1.x() - position_c.x()) ** 2 + (
                                        self.position1.y() - position_c.y()) ** 2) ** .5
                            if length > length_c:
                                length = length_c
                                self.cellId_e = itemID
                                self.cellName_e = nom
                    if length > 100:
                        self.linkattente = False  # 在想新建链接时，通过遍历找到与点击位置距离最近的神经集群或者噪声源进行连接
                    else:
                        index_item = findID(self.parent.Graph_Items, self.cellId_e)
                        self.position1 = self.parent.Graph_Items[index_item].dictionnaire["Pos"]
                        self.parent.textBrowser.setText("choose the emitting cell")
                        self.parent.textBrowser.append(
                            "\n You selected the cell : " + self.parent.Graph_Items[index_item].dictionnaire["Name"])
                        self.parent.textBrowser.append("\n choose the receving cell")
                elif self.linkattente == True:
                    self.linkattente = False
                    position2 = QPointF(event.pos())
                    position2 = self.mapToScene(position2.x(), position2.y())
                    cellId_r = -1
                    cellName_r = ''
                    length = np.inf
                    for idx, item in enumerate(self.parent.Graph_Items):
                        itemID = item.dictionnaire["ID"]
                        type = item.dictionnaire["Type"]
                        nom = item.dictionnaire["Name"]
                        position_c = item.dictionnaire["Pos"]
                        if type in range(-99, -1) or type in [-100]:
                            length_c = ((position2.x() - position_c.x()) ** 2 + (
                                    position2.y() - position_c.y()) ** 2) ** .5
                            if length > length_c:
                                length = length_c
                                cellId_r = itemID
                                cellName_r = nom
                    index_item = findID(self.parent.Graph_Items, cellId_r)
                    position2 = self.parent.Graph_Items[index_item].dictionnaire["Pos"]
                    self.parent.textBrowser.append(
                        "\n You selected the cell : " + self.parent.Graph_Items[index_item].dictionnaire["Name"])

                    if self.parent.Graph_Items[index_item].dictionnaire['Type'] == -100:
                        msg_cri('Noise object cannot receive signal')
                        self.parent.Add.setChecked(0)
                        return

                    if cellId_r == self.cellId_e:
                        Pos_c = QPointF(self.position1.x() + 100, self.position1.y() + 100)
                    else:
                        Pos_c = QPointF((position2.x() - self.position1.x()) * 0.5 + self.position1.x(),
                                        (position2.y() - self.position1.y()) * 0.5 + self.position1.y())

                    index_item = findID(self.parent.Graph_Items, self.cellId_e)

                    if self.parent.Graph_Items[index_item].dictionnaire['Type'] == -100:
                        Cx = ''
                    else:
                        Cx = self.parent.Cx_edit.text().replace(',', '.')

                    self.parent.Graph_Items.append(new_item_to_plot(ID=ID,
                                                             Type=self.parent.number_group.checkedId(),
                                                             Name=self.cellName_e + '_to_' + cellName_r,
                                                             Pos=self.position1, Pos2=position2,
                                                             Pos_c=Pos_c,
                                                             cellId_e=self.cellId_e, cellId_r=cellId_r,
                                                             Cx=Cx))
                    self.parent.forundo()  # ？？？？？
                    self.parent.Add.setChecked(0)
                    self.UpdateScene()
            self.clickedonitem = -1  # ?????

        elif self.parent.Rem.isChecked() == True or event.modifiers() == Qt.ShiftModifier or rightclickinfp == 'Remove':

            position = QPointF(event.pos())
            position = self.mapToScene(int(position.x()), int(position.y()))
            # print(position)
            findellipse = False
            findellipse_pos = []
            for item in self.scene.items():  # seek for ellipse                      # 得到场景中的所有图形项，并进行遍历
                if item.type() == QGraphicsEllipseItem().type():  # 获取椭圆形状图形项的类型标识
                    # print(item.contains(position))
                    if item.contains(position):  # 检查该图形项是否包含给定的位置
                        # print('find')
                        # self.scene.removeItem(item)
                        findellipse = True
                        findellipse_pos = item.boundingRect()  # 将该图形项的边界矩形赋值给 findellipse_pos 变量
                        # print(findellipse_pos)
                        break
            if findellipse == True:
                todelet = []
                for idx, item in enumerate(self.parent.Graph_Items):
                    type = item.dictionnaire["Type"]
                    if type in range(-99, -1) or type in [-100]:
                        position_c = item.dictionnaire["Pos"]
                        if position_c == findellipse_pos.center():  # 精确定位
                            # print('find')
                            itemid = item.dictionnaire["ID"]
                            Name = item.dictionnaire["Name"]
                            for idx2, item2 in enumerate(self.parent.Graph_Items):  # delet assiciated link
                                type2 = item2.dictionnaire["Type"]
                                if type2 in [-200]:
                                    if item2.dictionnaire["cellId_e"] == itemid:
                                        todelet.append(idx2)
                                    elif item2.dictionnaire["cellId_r"] == itemid:
                                        todelet.append(idx2)  # 将想要删除的神经集群所相连的链接的索引记录在 todelet 中

                            todelet.append(idx)
                            for todel in reversed(sorted(todelet)):  # 使用 reversed() 对 sorted(todelet) 列表进行反向迭代，目的是从后往前删除索引，避免在删除过程中索引的变化导致错误。
                                del self.parent.Graph_Items[todel]

            else:  # rechercher supprimer la link la plus proche（查找并删除最近的链接）  # 如果点击事件的位置不在椭圆形形状图形项内部
                cellId = -1
                length = np.inf
                for idx, item in enumerate(self.parent.Graph_Items):
                    # print(item)
                    type = item.dictionnaire["Type"]
                    if type in [-200]:
                        position_c = item.dictionnaire["Pos_c"]
                        length_c = ((position.x() - position_c.x()) ** 2 + (position.y() - position_c.y()) ** 2) ** .5
                        if length > length_c:
                            length = length_c
                            cellId = idx  # 找出距离点击位置最近的链接
                if length < 100 and not (cellId == -1):  # 如果距离够小
                    del self.parent.Graph_Items[cellId]
            self.UpdateScene()
            self.parent.Rem.setChecked(0)
            self.parent.forundo()
            self.clickedonitem = -1  # ?????

        elif self.parent.Mod.isChecked() == True or event.modifiers() == Qt.ControlModifier or rightclickinfp == 'Modify':

            position = QPointF(event.pos())
            position = self.mapToScene(int(position.x()), int(position.y()))
            # print(position)
            findellipse = False
            findellipse_pos = []
            for item in self.scene.items():  # seek for ellipse
                if item.type() == QGraphicsEllipseItem().type():
                    # print(item.contains(position))
                    if item.contains(position):
                        # self.scene.removeItem(item)
                        findellipse = True
                        findellipse_pos = item.boundingRect()
                        # print(findellipse_pos)
                        break
            if findellipse == True:  # e
                # print('true')
                for idx, item in enumerate(self.parent.Graph_Items):
                    itemid = item.dictionnaire["ID"]
                    type = item.dictionnaire["Type"]
                    Name = item.dictionnaire["Name"]
                    if type in range(-99, -1) or type in [-100]:
                        position_c = item.dictionnaire["Pos"]
                        if position_c == findellipse_pos.center():
                            self.parent.Mod.setChecked(0)
                            popNames = getNames(self.parent.Graph_Items)
                            exPopup = Questionparam(self, item=item, Graph_Items=self.parent.Graph_Items)

                            if exPopup.exec_() == QDialog.Accepted:
                                if not (exPopup.item.dictionnaire['Name'] == Name):  # 如果修改了集群或噪声的名字

                                    if exPopup.item.dictionnaire['Name'] in popNames:  # 检查修改后的名称是否已经存在于 popNames 列表中。如果存在，进入循环
                                        while exPopup.item.dictionnaire['Name'] in popNames:
                                            text, ok = QInputDialog.getText(self, '', "The  Name '" +
                                                                            exPopup.item.dictionnaire['Name'] +
                                                                            "' Already exists\n Please, enter another one:")
                                            if ok:
                                                exPopup.item.dictionnaire[
                                                    'Name'] = text  # 如果用户输入了新的名称，则将 exPopup.item 的名称更新为新的名称，循环会一直执行，直到用户提供了一个不重复的名称为止。

                                self.parent.Graph_Items[idx] = exPopup.item  # 如果名字没问题了，将 Graph_Items 中该 item 的所有信息替换

                                for idx2, item2 in enumerate(self.parent.Graph_Items):  # delet assiciated link 将对应的链接名称进行修改
                                    type2 = item2.dictionnaire["Type"]
                                    if type2 in [-200]:
                                        if item2.dictionnaire["cellId_e"] == itemid or item2.dictionnaire["cellId_r"] == itemid:
                                            cellName_e = self.parent.Graph_Items[
                                                findID(self.parent.Graph_Items, item2.dictionnaire["cellId_e"])].dictionnaire[
                                                'Name']
                                            cellName_r = self.parent.Graph_Items[
                                                findID(self.parent.Graph_Items, item2.dictionnaire["cellId_r"])].dictionnaire[
                                                'Name']
                                            self.parent.Graph_Items[idx2].dictionnaire[
                                                'Name'] = cellName_e + '_to_' + cellName_r

                                print('Accepted')
                            else:
                                print('Cancelled')
                            exPopup.deleteLater()

            else:  # rechercher et supprimer la link la plus proche
                cellId = -1
                length = np.inf
                for idx, item in enumerate(self.parent.Graph_Items):
                    # print(item)
                    itemid = item.dictionnaire["ID"]
                    type = item.dictionnaire["Type"]
                    if type in [-200]:
                        position_c = item.dictionnaire["Pos_c"]
                        length_c = ((position.x() - position_c.x()) ** 2 + (position.y() - position_c.y()) ** 2) ** .5
                        if length > length_c:
                            length = length_c
                            cellId = idx
                if length < 100:
                    self.parent.Mod.setChecked(0)
                    exPopup = Questionparam(self, item=self.parent.Graph_Items[cellId], Graph_Items=self.parent.Graph_Items)
                    # exPopup.show()
                    if exPopup.exec_() == QDialog.Accepted:
                        # exPopup.editfromlabel()
                        self.parent.Graph_Items[cellId] = exPopup.item
                        print('Accepted')
                    else:
                        print('Cancelled')
                    exPopup.deleteLater()
            self.clickedonitem = -1
            self.UpdateScene()
            self.parent.forundo()

        else:  # 如果只是瞎点
            cellId = -1
            position = QPointF(event.pos())
            position = self.mapToScene(int(position.x()), int(position.y()))
            # position = self.mapToScene(int(position.x()), int(position.y()))
            # print(position1)
            findellipse = False
            findellipse_pos = []
            for item in self.scene.items():  # seek for ellipse
                if item.type() == QGraphicsEllipseItem().type():
                    # print(item.contains(position1))
                    if item.contains(position):  # 如果点到的位置包含在了集群或者噪声里
                        # self.scene.removeItem(item)
                        findellipse = True
                        findellipse_pos = item.boundingRect()
                        # print(findellipse_pos)
                        break
            if findellipse == True:  # 如果 item 是神经集群或者噪声源，打印它的信息
                for idx, item in enumerate(self.parent.Graph_Items):
                    type = item.dictionnaire["Type"]
                    if type in range(-99, -1) or type in [-100]:
                        position_c = item.dictionnaire["Pos"]
                        if position_c == findellipse_pos.center():
                            self.parent.printItem(item)
                            cellId = idx
            else:  # rechercher et supprimer la link la plus proche
                cellId = -1
                length = np.inf
                for idx, item in enumerate(self.parent.Graph_Items):
                    # print(item)
                    type = item.dictionnaire["Type"]
                    if type in [-200]:
                        position_c = item.dictionnaire["Pos_c"]
                        length_c = ((position.x() - position_c.x()) ** 2 + (position.y() - position_c.y()) ** 2) ** .5
                        if length > length_c:
                            length = length_c
                            cellId = idx
                if length < 100:
                    self.parent.printItem(self.parent.Graph_Items[cellId])  # 打印离点击位置最近的链接信息
                else:
                    cellId = -1
                    self.positionclickedon = self.mapToScene(event.pos())
            self.clickedonitem = cellId

    def wheelEvent(self, event):
        if event.angleDelta().y() > 0:
            factor = self.parent.facteurzoom
        else:
            factor = 1 - (self.parent.facteurzoom - 1)
        self.scale(factor, factor)
        self.scene.update()

    def UpdateScene(self, ):
        for item in self.scene.items():
            self.scene.removeItem(item)

        for item in self.parent.Graph_Items:  # 遍历所有图形项，应当分为神经集群（附带噪声源）和链接
            itemid = item.dictionnaire["ID"]
            type = item.dictionnaire["Type"]
            nom = item.dictionnaire["Name"]
            position = item.dictionnaire["Pos"]
            if type == -200:  # interneuron inhibiteur                                  # 如果是链接，找它的左端神经元 index,与左端神经元的颜色统一
                index_item = findID(self.parent.Graph_Items, item.dictionnaire["cellId_e"])
                color = self.parent.Graph_Items[index_item].dictionnaire['color']
                color = QColor(color)
                type = item.dictionnaire["Type"]
                nom = item.dictionnaire["Name"]
                position2 = item.dictionnaire["Pos2"]
                position_c = item.dictionnaire["Pos_c"]
                Cx = item.dictionnaire["Cx"]

                xsize = 10
                percent = 0

                if item.dictionnaire["cellId_e"] == item.dictionnaire["cellId_r"]:  # 如果该链接的出发点与终点相同，该链接的画图流程，直接cv吧

                    midpoint = (position + position_c) / 2

                    rx = abs(midpoint.x() - position.x())
                    if rx <= 1.:
                        rx = 1.
                    ry = abs(midpoint.y() - position.y())
                    if ry <= 1.:
                        ry = 1.
                    r = (rx ** 2 + ry ** 2) ** 0.5
                    # print((position.x(),position.y()),(rx, ry, r),(position.x(),position.y()-  r),(position.x() + 2*r ,position.y()+  r))
                    path = QPainterPath()
                    path.addEllipse(QPointF(midpoint.x(), midpoint.y()), rx * (r / rx), ry * (r / ry))
                    # rot
                    rx = midpoint.x() - position.x()
                    # if rx<=1.:
                    #     rx=1.
                    ry = midpoint.y() - position.y()
                    # if ry<=1.:
                    #     ry=1.
                    phi1 = [round(np.mod(np.arcsin(ry / r), 2 * np.pi), 3),
                            round(np.mod(np.pi - np.arcsin(ry / r), 2 * np.pi), 3)]
                    phi2 = [round(np.mod(np.arccos(rx / r), 2 * np.pi), 3),
                            round(np.mod(2 * np.pi - np.arccos(rx / r), 2 * np.pi), 3)]

                    phi1 = [np.arcsin(ry / r), np.pi - np.arcsin(ry / r)]
                    for idx, val in enumerate(phi1):
                        if val < 0:
                            phi1[idx] = val + 2 * np.pi
                        phi1[idx] = round(phi1[idx], 1)
                    phi2 = [np.arccos(rx / r), 2 * np.pi - np.arccos(rx / r)]
                    for idx, val in enumerate(phi2):
                        if val < 0:
                            phi2[idx] = val + 2 * np.pi
                        phi2[idx] = round(phi2[idx], 1)

                    if phi1[0] == phi2[0]:
                        phi = phi1[0]
                    elif phi1[0] == phi2[1]:
                        phi = phi1[0]
                    elif phi1[1] == phi2[0]:
                        phi = phi1[1]
                    elif phi1[1] == phi2[1]:
                        phi = phi1[1]
                    else:
                        phi = 0
                    t = QTransform()

                    t.translate(path.boundingRect().center().x(), path.boundingRect().center().y())
                    t.rotate(phi * 180 / np.pi)
                    t.translate(-path.boundingRect().center().x(), -path.boundingRect().center().y())

                    # t.rotate(phi*180/np.pi)
                    path = t.map(path)

                    # path2 = QPainterPath()
                    # pointmidle = path.pointAtPercent(0)
                    # path2.moveTo(pointmidle)
                    # rect = newtext.boundingRect()
                    # newtext.setPos(position_c.x() - rect.width()/2  , position_c.y()-rect.height()/2 )
                    # path2.lineTo(position_c.x() , position_c.y())
                    # self.scene.addPath(path2,QPen(QColor(128, 128, 128, 128),2))

                    arrow = QPolygonF([QPointF(0, 20), QPointF(0, -20), QPointF(20, 0)])
                    t = QTransform()
                    t.rotate(-path.angleAtPercent(0))
                    # t.translate(path.pointAtPercent(0.75))
                    arrow = t.map(arrow)
                    arrow.translate(path.pointAtPercent(0))
                    self.scene.addPolygon(arrow, QPen(color, 5), QBrush(color))

                    self.scene.addPath(path, QPen(color, 5))
                    if self.withouttext:  # without text =0
                        newtext = QGraphicsTextItem(nom + '\nC:' + Cx)
                        font = QFont()
                        font.setPixelSize(18)
                        newtext.setFont(font)
                        rect = newtext.boundingRect()
                        newtext.setPos(position_c.x() - rect.width() / 2, position_c.y() - rect.height() / 2)
                        self.scene.addItem(newtext)
                else:
                    path = QPainterPath()
                    path.moveTo(position.x(), position.y())  # 将路径的起始点设置为 position 的坐标
                    # midpoint = (position + position2) / 2
                    midpoint = (position + position2) / 2
                    x = 2 * (position_c.x() - midpoint.x()) + midpoint.x()
                    y = 2 * (position_c.y() - midpoint.y()) + midpoint.y()
                    pos = QPointF(x, y)

                    path.quadTo(pos.x(), pos.y(), position2.x(), position2.y())  # quadTo() 包含控制点和终点，绘制平滑贝塞尔曲线
                    self.scene.addPath(path, QPen(color, 5))

                    # path2 = QPainterPath()
                    # pointmidle = path.pointAtPercent(0.5)
                    # path2.moveTo(pointmidle)
                    # rect = newtext.boundingRect()
                    # newtext.setPos(position_c.x() - rect.width()/2  , position_c.y()-rect.height()/2 )
                    # path2.lineTo(position_c.x() , position_c.y())
                    # self.scene.addPath(path2,QPen(QColor(128, 128, 128, 128),2))

                    arrow = QPolygonF([QPointF(0, 20), QPointF(0, -20), QPointF(20, 0)])
                    t = QTransform()
                    t.rotate(-path.angleAtPercent(0.75))  # 旋转变换？
                    # t.translate(path.pointAtPercent(0.75))
                    arrow = t.map(arrow)
                    ab = 1 - 150. / ((item.dictionnaire['Pos'] - item.dictionnaire['Pos_c']).manhattanLength() +
                                     (item.dictionnaire['Pos_c'] - item.dictionnaire['Pos2']).manhattanLength())
                    # Pos_1 = QPointF(item.dictionnaire['Pos'][0], item.dictionnaire['Pos'][1])
                    # Pos_2 = QPointF(item.dictionnaire['Pos2'][0], item.dictionnaire['Pos2'][1])
                    # Pos_c = QPointF(item.dictionnaire['Pos_c'][0], item.dictionnaire['Pos_c'][1])
                    # ab = 1 - 150. / ((Pos_1 - Pos_c).manhattanLength() + (Pos_c - Pos_2).manhattanLength())

                    arrow.translate(path.pointAtPercent(0.75))  # 坐标变换
                    if self.withouttext:  # without text =0
                        newtext = QGraphicsTextItem(nom + '\nC:' + Cx)
                        font = QFont()
                        font.setPixelSize(20)
                        newtext.setFont(font)
                        rect = newtext.boundingRect()
                        newtext.setPos(position_c.x() - rect.width() / 2, position_c.y() - rect.height() / 2)
                        self.scene.addItem(newtext)  # 添加该文本项

                    self.scene.addPolygon(arrow, QPen(color, 5), QBrush(color))

        for item in self.parent.Graph_Items:
            type = item.dictionnaire["Type"]
            nom = item.dictionnaire["Name"]
            position = item.dictionnaire["Pos"]
            Hx = item.dictionnaire["Hx"]
            lamda = item.dictionnaire["lamda"]

            if type in range(-99, -1) or type in [-100]:
                color = item.dictionnaire["color"]
                color = QColor(color)

                xsize = 120
                ysize = 120
                newitem = QGraphicsEllipseItem(position.x() - xsize / 2., position.y() - ysize / 2., xsize, ysize)
                newitem.setBrush(QBrush(color, style=Qt.SolidPattern))
                self.scene.addItem(newitem)

                if type == -100:
                    mean = item.dictionnaire["Noi_mean"]
                    std = item.dictionnaire["Noi_std"]
                    # gt = item.dictionnaire["noisegainT"]
                    # newtext = QGraphicsTextItem(nom + '\nm:' + mean + '\ns:' + std + '\ngain:' + gt)
                    newtext = QGraphicsTextItem(nom + '\nm:' + mean + '\ns:' + std)
                else:
                    newtext = QGraphicsTextItem(nom + '\nH:' + Hx + '\nλ:' + lamda)
                # doc = QTextDocument()
                # doc.setDefaultTextOption(QTextOption(Qt.AlignCenter))
                # newtext.setDocument(doc)
                # textItem.document()->setDefaultTextOption(QTextOption(Qt::AlignCenter | Qt::AlignVCenter))
                if self.withouttext:  # without text =0
                    font = QFont()
                    font.setPixelSize(15)
                    newtext.setFont(font)
                    rect = newtext.boundingRect()
                    newtext.setPos(position.x() - rect.width() / 2, position.y() - rect.height() / 2)
                    self.scene.addItem(newtext)


class new_item_to_plot:
    def __init__(self, ID=None, Type=None, Name=None, Pos=None, Pos2=None, Pos_c=None, cellId_e=None, cellId_r=None,
                 Hx=None, lamda=None, Cx=None, e0=None, v0=None, rx=None, Noi_mean=None, Noi_std=None, color=None):
        self.dictionnaire = dict()
        self.dictionnaire["ID"] = ID
        self.dictionnaire["Type"] = Type
        self.dictionnaire["Name"] = Name
        self.dictionnaire["Pos"] = Pos
        self.dictionnaire["Pos2"] = Pos2
        self.dictionnaire["Pos_c"] = Pos_c
        self.dictionnaire["cellId_e"] = cellId_e
        self.dictionnaire["cellId_r"] = cellId_r
        self.dictionnaire["color"] = color
        self.dictionnaire["e0"] = e0
        self.dictionnaire["v0"] = v0
        self.dictionnaire["rx"] = rx
        self.dictionnaire["Hx"] = Hx
        self.dictionnaire["lamda"] = lamda
        self.dictionnaire["Cx"] = Cx
        self.dictionnaire["Noi_mean"] = Noi_mean
        self.dictionnaire["Noi_std"] = Noi_std
        self.dictionnaire["eq1"] = None
        self.dictionnaire["eq2"] = None
        self.dictionnaire["lfp"] = None
        self.dictionnaire["PSP"] = None
        self.dictionnaire["FR"] = None
        self.dictionnaire["noise"] = None
        self.dictionnaire["linknoise_r"] = None
        self.dictionnaire["linkId_r"] = None
        self.dictionnaire["Hname"] = None
        self.dictionnaire["Tname"] = None
        self.dictionnaire["Cname"] = None
        self.dictionnaire["e0name"] = None
        self.dictionnaire["v0name"] = None
        self.dictionnaire["rname"] = None
        self.dictionnaire["meanname"] = None
        self.dictionnaire["stdname"] = None
        self.dictionnaire["sigmname"] = None
        self.dictionnaire["noisename"] = None
        self.dictionnaire["noisevarname"] = None
        self.dictionnaire["noisegainT"] = None
        self.dictionnaire["pulseName"] = None
        self.dictionnaire["ppsName"] = None


class Questionparam(UI_modify):
    def __init__(self, parent=None, item=None, Graph_Items=None):
        super().__init__()
        self.parent = parent
        self.Graph_Items = Graph_Items
        self.item = item
        self.color_Button.clicked.connect(self.change_color)
        self.button_color = None

        label = ['Name', 'e0', 'v0', 'rx', 'Hx', 'lamda', 'Cx', 'Noi_Mean', 'Noi_std', 'color']
        edit = self.labelfromedit()
        type = item.dictionnaire['Type']

        if type in range(-99, -1):
            self.Name_edit.setDisabled(0)
            self.e0_edit.setDisabled(0)
            self.v0_edit.setDisabled(0)
            self.rx_edit.setDisabled(0)
            self.Hx_edit.setDisabled(0)
            self.lamda_edit.setDisabled(0)
            self.Cx_edit.setDisabled(1)
            self.Noi_mean_edit.setDisabled(1)
            self.Noi_std_edit.setDisabled(1)
            # self.noise_GTe.setDisabled(1)
            self.color_Button.setDisabled(0)
        elif type in [-200]:
            self.Name_edit.setDisabled(1)
            self.e0_edit.setDisabled(1)
            self.v0_edit.setDisabled(1)
            self.rx_edit.setDisabled(1)
            self.Hx_edit.setDisabled(1)
            self.lamda_edit.setDisabled(1)
            self.Cx_edit.setDisabled(0)
            self.Noi_mean_edit.setDisabled(1)
            self.Noi_std_edit.setDisabled(1)
            # self.noise_GTe.setDisabled(0)
            self.color_Button.setDisabled(1)
        elif type in [-100]:
            self.Name_edit.setDisabled(0)
            self.e0_edit.setDisabled(1)
            self.v0_edit.setDisabled(1)
            self.rx_edit.setDisabled(1)
            self.Hx_edit.setDisabled(1)
            self.lamda_edit.setDisabled(1)
            self.Cx_edit.setDisabled(1)
            self.Noi_mean_edit.setDisabled(0)
            self.Noi_std_edit.setDisabled(0)
            self.color_Button.setDisabled(0)

        self.Name_edit.setText(self.item.dictionnaire['Name'])
        self.e0_edit.setText(self.item.dictionnaire['e0'])
        self.v0_edit.setText(self.item.dictionnaire['v0'])
        self.rx_edit.setText(self.item.dictionnaire['rx'])
        self.Hx_edit.setText(self.item.dictionnaire['Hx'])
        self.lamda_edit.setText(self.item.dictionnaire['lamda'])
        self.Cx_edit.setText(self.item.dictionnaire['Cx'])
        self.Noi_mean_edit.setText(self.item.dictionnaire['Noi_mean'])
        self.Noi_std_edit.setText(self.item.dictionnaire['Noi_std'])
        self.color_Button.setStyleSheet(f"background-color: {self.item.dictionnaire['color']}")

        self.pushButton_ok.clicked.connect(self.myaccept)
        self.pushButton_cancel.clicked.connect(self.reject)

    def myaccept(self):
        self.editfromlabel()
        self.accept()

    def labelfromedit(self):
        label = [self.item.dictionnaire['Name'],
                 self.item.dictionnaire['e0'],
                 self.item.dictionnaire['v0'],
                 self.item.dictionnaire['rx'],
                 self.item.dictionnaire['Hx'],
                 self.item.dictionnaire['lamda'],
                 self.item.dictionnaire['Cx'],
                 self.item.dictionnaire['Noi_mean'],
                 self.item.dictionnaire['Noi_std'],
                 # [self.item.dictionnaire["noisegainT"], findPopNale(self.Graph_Items)],
                 self.item.dictionnaire['color']]
        return label

    def change_color(self):
        color = QtWidgets.QColorDialog.getColor()
        if color.isValid():
            self.button_color = color.name()
            self.color_Button.setStyleSheet(f"background-color: {color.name()}")

    def editfromlabel(self):
        self.item.dictionnaire['Name'] = self.Name_edit.text()
        self.item.dictionnaire['e0'] = self.e0_edit.text().replace(',', '.')
        self.item.dictionnaire['v0'] = self.v0_edit.text().replace(',', '.')
        self.item.dictionnaire['rx'] = self.rx_edit.text().replace(',', '.')
        self.item.dictionnaire['Hx'] = self.Hx_edit.text().replace(',', '.')
        self.item.dictionnaire['lamda'] = self.lamda_edit.text().replace(',', '.')
        self.item.dictionnaire['Cx'] = self.Cx_edit.text().replace(',', '.')
        self.item.dictionnaire['Noi_mean'] = self.Noi_mean_edit.text().replace(',', '.')
        self.item.dictionnaire['Noi_std'] = self.Noi_std_edit.text().replace(',', '.')
        # self.item.dictionnaire["noisegainT"] = self.noise_GTe.currentText()
        # self.item.dictionnaire["color"] = QColor(self.Colorselection.palette().button().color())
        self.item.dictionnaire["color"] = self.button_color


class InfoTable(QDialog):
    def __init__(self, parent=None, Graph_Items=None):
        super(InfoTable, self).__init__(parent)
        self.parent = parent
        self.Graph_Items = Graph_Items
        self.setMinimumSize(800, 600)
        self.Param_box = QWidget()
        self.layout_Param_box = QVBoxLayout()

        self.tableWidget = QTableWidget()
        self.tableWidget.setRowCount(len(Graph_Items) + 10)
        nbcol = 7
        self.tableWidget.setColumnCount(nbcol)
        self.tableWidget.setSpan(0, 0, 1, nbcol)

        # population
        self.tableWidget.setItem(0, 0, QTableWidgetItem("Populations:"))
        col = 0
        raw = 1
        self.tableWidget.setItem(raw, col, QTableWidgetItem("Name"))
        col += 1
        self.tableWidget.setItem(raw, col, QTableWidgetItem("Hx"))
        col += 1
        self.tableWidget.setItem(raw, col, QTableWidgetItem("lamda"))
        col += 1
        self.tableWidget.setItem(raw, col, QTableWidgetItem("e0"))
        col += 1
        self.tableWidget.setItem(raw, col, QTableWidgetItem("v0"))
        col += 1
        self.tableWidget.setItem(raw, col, QTableWidgetItem("rx"))
        col += 1
        self.tableWidget.setItem(raw, col, QTableWidgetItem("color"))

        for item in Graph_Items:
            if item.dictionnaire["Type"] in range(-99, -1):
                col = 0
                raw += 1
                self.tableWidget.setItem(raw, col, QTableWidgetItem(str(item.dictionnaire["Name"])))
                col += 1
                self.tableWidget.setItem(raw, col, QTableWidgetItem(str(item.dictionnaire["Hx"])))
                col += 1
                self.tableWidget.setItem(raw, col, QTableWidgetItem(str(item.dictionnaire["lamda"])))
                col += 1
                self.tableWidget.setItem(raw, col, QTableWidgetItem(str(item.dictionnaire["e0"])))
                col += 1
                self.tableWidget.setItem(raw, col, QTableWidgetItem(str(item.dictionnaire["v0"])))
                col += 1
                self.tableWidget.setItem(raw, col, QTableWidgetItem(str(item.dictionnaire["rx"])))
                col += 1
                colorbut = QPushButton()
                # self.colorbut.clicked.connect(self.change_color)
                colorbut.setStyleSheet(f"background-color: {item.dictionnaire['color']}")
                # set_QPushButton_background_color(colorbut, item.dictionnaire["color"])
                self.tableWidget.setCellWidget(raw, col, colorbut)

        # Noises
        col = 0
        raw += 1
        self.tableWidget.setSpan(raw, col, 1, nbcol)
        raw += 1
        self.tableWidget.setSpan(raw, col, 1, nbcol)
        self.tableWidget.setItem(raw, col, QTableWidgetItem("Noises:"))
        raw += 1
        self.tableWidget.setItem(raw, col, QTableWidgetItem("Name"))
        col += 1
        self.tableWidget.setItem(raw, col, QTableWidgetItem("Noi_mean"))
        col += 1
        self.tableWidget.setItem(raw, col, QTableWidgetItem("Noi_std"))
        # col += 1
        # self.tableWidget.setItem(raw, col, QTableWidgetItem("noisegainT"))
        col += 1
        self.tableWidget.setItem(raw, col, QTableWidgetItem("color"))

        for item in Graph_Items:
            if item.dictionnaire["Type"] == -100:
                col = 0
                raw += 1
                self.tableWidget.setItem(raw, col, QTableWidgetItem(str(item.dictionnaire["Name"])))
                col += 1
                self.tableWidget.setItem(raw, col, QTableWidgetItem(str(item.dictionnaire["Noi_mean"])))
                col += 1
                self.tableWidget.setItem(raw, col, QTableWidgetItem(str(item.dictionnaire["Noi_std"])))
                # col += 1
                # self.tableWidget.setItem(raw, col, QTableWidgetItem(str(item.dictionnaire["noisegainT"])))
                col += 1
                colorbut = QPushButton()
                colorbut.setStyleSheet(f"background-color: {item.dictionnaire['color']}")
                # set_QPushButton_background_color(colorbut, item.dictionnaire["color"])
                self.tableWidget.setCellWidget(raw, col, colorbut)

        # Noises
        col = 0
        raw += 1
        self.tableWidget.setSpan(raw, col, 1, nbcol)
        raw += 1
        self.tableWidget.setSpan(raw, col, 1, nbcol)
        self.tableWidget.setItem(raw, col, QTableWidgetItem("Links:"))
        col = 0
        raw += 1
        self.tableWidget.setItem(raw, col, QTableWidgetItem("Name"))
        col += 1
        self.tableWidget.setItem(raw, col, QTableWidgetItem("Cx"))

        for item in Graph_Items:
            if item.dictionnaire["Type"] == -200:
                col = 0
                raw += 1
                self.tableWidget.setItem(raw, col, QTableWidgetItem('C_' + str(item.dictionnaire["Name"])))
                color_e = QColor(Graph_Items[findID(Graph_Items, item.dictionnaire['cellId_e'])].dictionnaire['color'])
                color_r = QColor(Graph_Items[findID(Graph_Items, item.dictionnaire['cellId_r'])].dictionnaire['color'])

                gradient = QLinearGradient(0, 0, 100, 0)
                gradient.setColorAt(0.0, color_e)
                gradient.setColorAt(1.0, color_r)
                brush = QBrush(gradient)

                self.tableWidget.item(raw, col).setBackground(brush)

                col += 1
                self.tableWidget.setItem(raw, col, QTableWidgetItem(str(item.dictionnaire["Cx"])))

        scroll = QScrollArea()
        widget = QWidget(self)
        widget.setLayout(QHBoxLayout())
        widget.layout().addWidget(self.tableWidget)
        scroll.setWidget(widget)
        scroll.setWidgetResizable(True)

        self.Button_Ok = QPushButton('Close')
        self.Button_Ok.setFixedSize(66, 30)
        self.Button_Ok.clicked.connect(self.accept)

        self.layout_Param_box.addWidget(scroll)

        self.layout_Param_box.addWidget(self.Button_Ok)
        self.Param_box.setLayout(self.layout_Param_box)
        self.setLayout(self.layout_Param_box)


def findID(Graph_Items=None, ID=None):
    for idx, item in enumerate(Graph_Items):
        if item.dictionnaire['ID'] == ID:
            return idx


def getNames(Graph_Items=None):
    Names = []
    for idx, item in enumerate(Graph_Items):
        if item.dictionnaire['Type'] in range(-100, -1):
            Names.append(item.dictionnaire['Name'])
    return Names


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    my_pyqt_form = Main_Window()
    my_pyqt_form.show()
    sys.exit(app.exec_())
