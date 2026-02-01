import random
import numpy as np

def get_LFP_Name():
    return ["LFP"]

def get_LFP_color():
    return ["#ff0000"]

def get_Pulse_Names():
    return ['FR_Exc','FR_P','FR_Inh',]

def get_PPS_Names():
    return ['PSP_Exc','PSP_P','PSP_Inh',]

def get_Colors():
    return ['#ffaa00','#ff0000','#0000ff',]

def get_ODE_solver():
    return ["derivT"]

def get_ODE_solver_Time():
    return ["deriv_Time"]

def get_Variable_Names():
    return ['H_EXC','T_EXC','e0_EXC','v0_EXC','r_EXC','H_P','T_P','e0_P','v0_P','r_P','H_INH','T_INH','e0_INH','v0_INH','r_INH','m_N','s_N','C_P_to_EXC','C_EXC_to_P','C_P_to_INH','C_INH_to_P',]

class Model:
    def __init__(self,):
        self.v0_EXC=6.0
        self.e0_EXC=5.0
        self.r_EXC=0.56
        self.H_EXC=3.25
        self.T_EXC=100.0
        self.FR_Exc= 0.
        self.PSP_Exc= 0.
        self.v0_P=6.0
        self.e0_P=5.0
        self.r_P=0.56
        self.H_P=3.25
        self.T_P=100.0
        self.FR_P= 0.
        self.PSP_P= 0.
        self.v0_INH=6.0
        self.e0_INH=5.0
        self.r_INH=0.56
        self.H_INH=22.0
        self.T_INH=50.0
        self.FR_Inh= 0.
        self.PSP_Inh= 0.
        self.m_N=220.0
        self.s_N=0
        self.noise_var_N=0.
        self.C_P_to_EXC=135.0
        self.C_EXC_to_P=108.0
        self.C_P_to_INH=33.75
        self.C_INH_to_P=33.75
        self.dt = 1./2048.
        self.NbODEs = 6
        self.init_vector( )

    def init_vector(self):
        self.dydt = np.zeros(self.NbODEs)
        self.y = np.zeros(self.NbODEs)
        self.yt = np.zeros(self.NbODEs)
        self.dydx1 = np.zeros(self.NbODEs)
        self.dydx2 = np.zeros(self.NbODEs)
        self.dydx3 = np.zeros(self.NbODEs)
        self.LFP = 0.

    def noise_N(self):
        return random.gauss(self.m_N,self.s_N)

    def sigm_EXC(self,v):
        return  self.e0_EXC/(1+np.exp(self.r_EXC*(self.v0_EXC-v)))

    def sigm_P(self,v):
        return  self.e0_P/(1+np.exp(self.r_P*(self.v0_P-v)))

    def sigm_INH(self,v):
        return  self.e0_INH/(1+np.exp(self.r_INH*(self.v0_INH-v)))

    def PTW(self,y0,y1,y2,V,v):
        return (V*v*y0 - 2*v*y2 - v*v*y1)

    def derivT(self):
        self.noise_var_N = self.noise_N()
        self.yt = self.y+0.
        self.dydx1=self.deriv()
        self.y = self.yt + self.dydx1 * self.dt / 2
        self.dydx2=self.deriv()
        self.y = self.yt + self.dydx2 * self.dt / 2
        self.dydx3=self.deriv()
        self.y = self.yt + self.dydx3 * self.dt
        self.y =self.yt + self.dt/6. *(self.dydx1+2*self.dydx2+2*self.dydx3+self.deriv())  # 1 2 3 4
        self.LFP =  + self.y[0] - self.y[4]
        self.PSP_Exc = self.y[0]
        self.FR_Exc =  + self.sigm_EXC(self.C_P_to_EXC * self.y[2])
        self.PSP_P = self.y[2]
        self.FR_P = self.sigm_P( + self.y[0] - self.y[4])
        self.PSP_Inh = self.y[4]
        self.FR_Inh =  + self.sigm_INH(self.C_P_to_INH * self.y[2])

    def deriv_Time(self,N):
        lfp = np.zeros(N,)
        for k in range(N):
            self.derivT()
            lfp[k]= self.LFP
        return lfp

    def deriv(self):
        self.dydt[0] = self.y[1]
        self.dydt[1] = self.PTW((self.noise_var_N+self.C_EXC_to_P* + self.sigm_EXC(self.C_P_to_EXC * self.y[2])), self.y[0],self.y[1],self.H_EXC , self.T_EXC)
        self.dydt[2] = self.y[3]
        self.dydt[3] = self.PTW((+self.sigm_P( + self.y[0] - self.y[4])), self.y[2],self.y[3],self.H_P , self.T_P)
        self.dydt[4] = self.y[5]
        self.dydt[5] = self.PTW((+self.C_INH_to_P* + self.sigm_INH(self.C_P_to_INH * self.y[2])), self.y[4],self.y[5],self.H_INH , self.T_INH)

        return self.dydt+0.
