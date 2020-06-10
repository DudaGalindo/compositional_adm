"""Check stability of a system and find its composition at a thermodynamic equilibrium."""
import numpy as np
from ..directories import data_loaded
from scipy.misc import derivative
from .equation_of_state_n import PengRobinson
from ..utils import constants as ctes
from numba import jit, jitclass, float64, int64
import math
import numba
# import matplotlib.pyplot as plt
## Encontrar os pontos estacionarios. Estes correspondem aos pontos nos quais a derivada de g com respeito a Y é 0
## Todas as equações foram confirmadas pelo livro do Dandekar e a biblioteca do thermo
# ph=0 - vapor. ph=1 - liquid.
global EOS_class


spec = [('P',float64), ('T', float64), ('w', float64[:]), ('Tc', float64[:]), ('Pc', float64[:]), ('Bin', float64[:,:]), ('Nc', int64), ('K', float64[:]), ('fv', float64[:]), ('fl', float64[:]), ('L', float64), ('V', float64), ('x', float64[:]), ('y', float64[:]), ('Pbubble', float64), ('ksi_L', float64), ('ksi_V', float64), ('rho_L', float64), ('rho_V', float64)]
@numba.experimental.jitclass(spec)
class StabilityCheck_n(object):
    """Check for stability of a thermodynamic equilibrium and returns the
    equilibrium phase compositions (perform the flash calculation)."""

    def __init__(self, P, T,  w, Tc, Pc, Bin):
        self.P = P
        self.T = T
        self.w = w
        self.Tc = Tc
        self.Pc = Pc
        self.Bin = Bin
        self.Nc = len(w)
        self.K = np.empty(self.Nc)
        self.fv = np.empty(self.Nc)
        self.fl = np.empty(self.Nc)
        self.L = 0.
        self.V = 0.
        self.x = np.ones(self.Nc)
        self.y = np.ones(self.Nc)
        self.Pbubble = 0.
        self.ksi_L = 0.
        self.ksi_V = 0.
        self.rho_L = 0.
        self.rho_V = 0.

    def run(self, z, Mw):
        self.equilibrium_ratio_Wilson()
        #getattr(equation_of_state_n, nameEOS)
        EOS = PengRobinson(self.P, self.T,  self.w, self.Tc, self.Pc, self.Bin)
        #lnphix = EOS.lnphi(self.x,0)
        if np.any(z<=0):
            self.molar_properties(EOS, z)
        else:
            sp1,sp2 = self.StabilityTest(EOS, z)
        if np.round(sp1,8) > 1 or np.round(sp2,8) > 1: self.molar_properties(EOS, z)
        else: #tiny manipulation
            self.x = z; self.y = z
            self.bubble_point_pressure()
            if self.P > self.Pbubble: self.L = 1; self.V = 0; ph = 1
            else: self.L = 0.; self.V = 1.; ph = 0.

        self.update_EOS_dependent_properties(EOS, Mw)


    def equilibrium_ratio_Wilson(self):
        self.K = np.exp(5.37 * (1 + self.w) * (1 - self.Tc / self.T)) * (self.Pc / self.P)



    """------------------- Stability test calculation -----------------------"""
    #@jit(nopython = True)
    def StabilityTest(self, EOS, z):
        ''' In the lnphi function: 0 stands for vapor phase and 1 for liquid '''
    #****************************INITIAL GUESS******************************#
    ## Both approaches bellow should be used in case the phase is in the critical region

    #*****************************Test one**********************************#
        #Used alone when the phase investigated (z) is clearly vapor like (ph = 0)

        Y = z / self.K
        Yold = 0.9 * Y
        y = Y / np.sum(Y)
        lnphiz = EOS.lnphi(z, 0)
        while np.max(np.abs(Y / Yold - 1)) > 1e-9: #convergência
            Yold = np.copy(Y)
            lnphiy = EOS.lnphi(y, 1)
            Y = np.exp(np.log(z) + lnphiz - lnphiy)
            y = Y / np.sum(Y)
        stationary_point1 = np.sum(Y)

    #*****************************Test two**********************************#
        #Used alone when the phase investigated (z) is clearly liquid like (ph == 1)

        Y = self.K * z
        Y_old = 0.9 * Y
        y = Y / np.sum(Y)
        lnphiz = EOS.lnphi(z, 1)
        while np.max(np.abs(Y / Y_old - 1)) > 1e-9:
            Y_old = np.copy(Y)
            lnphiy = EOS.lnphi(y, 0)
            Y = np.exp(np.log(z) + lnphiz - lnphiy)
            y = Y / np.sum(Y)
        stationary_point2 = np.sum(Y)

        return stationary_point1, stationary_point2


    """-------------------- Biphasic flash calculations ---------------------"""

    def molar_properties(self, EOS, z):
        self.fv = 2 * np.ones(len(z)); self.fl = np.ones(len(z)) #entrar no primeiro loop
        if self.Nc<= 2: self.molar_properties_Whitson(EOS, z)
        else: self.molar_properties_Yinghui(EOS, z)

    def deltaG_molar(self, EOS, l, ph):
        lnphi = np.empty(2).reshape(2,self.Nc)
        lnphi[0,:] = EOS.lnphi(l, 1 - ph)
        #lnphi[1,:] = EOS.lnphi(l, ph)
        #deltaG_molar = np.sum(l * (lnphi[1 - ph] - lnphi[ph]))
        #if deltaG_molar < 0: ph = 1 - ph
        return ph

    def deltaG_molar_vectorized(EOS, l, ph):
        lnphi = np.empty([2, len(ph)])
        lnphi[0,:] = EOS.lnphi(l, 1 - ph)
        lnphi[1,:] = EOS.lnphi(l, ph)

        deltaG_molar = np.sum(l * (lnphi[1 - ph] - lnphi[ph]), axis=0)
        ph[deltaG_molar<0] = 1 - ph[deltaG_molar<0]
        return ph

    def lnphi_based_on_deltaG(self, EOS, l, ph):
        #ph = np.array(ph)[:,np.newaxis]
        ph = self.deltaG_molar(EOS, l, ph)
        return EOS.lnphi(l, ph)

    #@jit(nopython = True)
    def solve_objective_function_Yinghui(self, z1, zi, z, K1, KNc, Ki):

        x1_min = z1[0] * (1. - KNc) / (K1 - KNc)

        x1_max = (1. - KNc) / (K1 - KNc)

        if np.any(z < 0):
            theta = np.ones(len(z))
            theta[self.K > 1] = (1 - KNc) / (self.K[self.K > 1] - KNc)
            aux_eq = (self.K - 1) * z1 / (z * (K1 - 1) / theta - (K1 - self.K))
            if np.all((self.K[z != 0] - 1) * z1 / z[z != 0] > 0):
                aux_eq = aux_eq[aux_eq >= 0] #se arr<0 é descartado
                x1_max = np.min(aux_eq)
            else:
                if np.max(aux_eq) > 0.: x1_min = np.max(aux_eq)
                else: x1_min = 0.

        #if x1_min.ravel() > x1_max.ravel(): raise ValueError('There is no physical root')

        x1 = (x1_min + x1_max) / 2.
        f = 1.

        while np.abs(f) > 1e-10:
            f = 1 + ((K1 - KNc) / (KNc - 1)) * x1 + np.sum(((Ki - KNc) / (KNc - 1))
               * zi * (K1 - 1) * x1 / ((Ki - 1) * z1 + (K1 - Ki) * x1))
            df = ((K1 - KNc) / (KNc - 1)) + np.sum(((Ki - KNc) / (KNc - 1)) * zi *
                z1 * (K1 - 1)* (Ki - 1) / ((Ki - 1) * z1 + (K1 - Ki) * x1) ** 2)
            x1 = x1 - f/df #Newton-Raphson iterative method
            if (x1 > x1_max) or (x1 < x1_min): x1 = (x1_min + x1_max)/2. #recommended
            if f * df > 0: x1_max = x1
            if f * df < 0: x1_min = x1

        xi = (K1 - 1.) * zi * x1 / ((Ki - 1.) * z1 + (K1 - Ki) * x1)
        self.x[self.K == K1] = x1
        self.x[self.K == KNc] = 1. - np.sum(xi) - x1

        return xi

    #@jit(nopython=True)
    def Yinghui_method(self, z):

        """ Shaping K to Nc-2 components by removing K1 and KNc """
        K1 = np.max(self.K); KNc = np.min(self.K)
        Ki = self.K[(self.K != K1) & (self.K != KNc)]

        """ Shaping z to Nc-2 components by removing z1 and zNc """
        z1 = z[self.K == K1]
        index1 = np.argwhere(self.K == K1).ravel()
        i = np.arange(self.Nc)
        zi1 = z[i != index1]

        indexNc = np.argwhere(self.K == KNc).ravel()
        i = np.arange(len(zi1))
        zi = zi1[i!=indexNc]

        #starting x
        # self.x = np.zeros(kprop.Nc)

        """ Solution """
        if z1[0] != 0.:
            xi = self.solve_objective_function_Yinghui(z1, zi, z, K1, KNc, Ki)
        else:
            x=2
            xi = (K1 - 1.) * zi / (K1 - Ki)
            self.x[self.K == KNc] = (K1 - 1.) * z[self.K == KNc] / (K1 - self.K[self.K == KNc])
            self.x[self.K == K1] = 1. - np.sum(xi) - np.sum(self.x)

        #ainda não sei como tirar esse for
        for j in range(len(xi)):
            self.x[self.K == Ki[j]] = xi[j]

        self.y = self.K * self.x

    #@jit(nopython=True)
    def molar_properties_Yinghui(self, EOS, z):
        #razao = fl/fv -> an arbitrary vector to enter in the iterative mode
        razao = np.ones(self.Nc)/2
        while np.max(np.abs(razao - 1)) > 1e-9:
            self.Yinghui_method(z)
            lnphil = self.lnphi_based_on_deltaG(EOS, self.x, 1)
            lnphiv = self.lnphi_based_on_deltaG(EOS, self.y, 0)
            self.fl = np.exp(lnphil) * (self.x * self.P)
            self.fv = np.exp(lnphiv) * (self.y * self.P)
            razao  = self.fl[self.fv!=0]/self.fv[self.fv!=0]*(1+1e-10)
            self.K = razao * self.K
        self.V = ((z[self.x != 0] - self.x[self.x != 0]) / (self.y[self.x != 0] - self.x[self.x != 0]))[0]

    #@jit(nopython=True)
    def solve_objective_function_Whitson(self, z):
        """ Solving for V """
        Vmax = 1 / (1 - np.min(self.K))
        Vmin = 1 / (1 - np.max(self.K))
        #Vmin = ((K1-KNc)*z[self.K==K1]-(1-KNc))/((1-KNc)*(K1-1))
        #proposed by Li et al for Whitson method
        self.V = (Vmin + Vmax) / 2
        Vold = self.V / 2 #just to get into the loop

        while np.abs(self.V / Vold - 1) > 1e-8:
            Vold = self.V
            f = np.sum((self.K - 1) * z / (1 + self.V * (self.K - 1)))
            df = -np.sum((self.K - 1) ** 2 * z / (1 + self.V * (self.K - 1)) ** 2)
            self.V = self.V - f / df #Newton-Raphson iterative method

            if self.V > Vmax: self.V = Vmax #(Vmax + Vold)/2
            elif self.V < Vmin: self.V = Vmin #(Vmin + Vold)/2

        self.x = z / (1 + self.V * (self.K - 1))
        self.y = self.K * self.x

    #@jit(nopython=True)
    def molar_properties_Whitson(self, EOS, z):
        razao = np.ones(self.Nc)/2
        while np.max(np.abs(self.fv / self.fl - 1)) > 1e-9:
            self.solve_objective_function_Whitson(z)
            lnphil = self.lnphi_based_on_deltaG(EOS, self.x, 1)
            lnphiv = self.lnphi_based_on_deltaG(EOS, self.y, 0)
            self.fv = np.exp(lnphiv) * (self.y * self.P)
            self.fl = np.exp(lnphil) * (self.x * self.P)
            razao  = self.fl[self.fv!=0]/self.fv[self.fv!=0]*(1+1e-10)
            #razao = np.floor_divide(self.fl, self.fv, out = out, where = True)
            self.K = razao * self.K

    def bubble_point_pressure(self):
        #Isso vem de uma junção da Lei de Dalton com a Lei de Raoult
        Pv = self.K * self.P
        self.Pbubble = np.sum(self.x * Pv)

    def update_EOS_dependent_properties(self, EOS, Mw):
        #EOS = EOS_class(self.P, fprop.T, kprop)
        ph_L = 1 #np.ones(ctes.n_volumes).astype(int)
        ph_V = 0 #np.zeros(ctes.n_volumes).astype(int)
        self.ksi_L, self.rho_L = self.get_EOS_dependent_properties(EOS, Mw, self.x, ph_L)
        self.ksi_V, self.rho_V = self.get_EOS_dependent_properties(EOS, Mw, self.y, ph_V)

    def get_EOS_dependent_properties(self, EOS, Mw, l, ph):
        #l - any phase molar composition
        # This won't work for everything, I'm sure of that. Still need, tho, to check the cases that this won't work.
        R = 8.3144598
        A, B = EOS.coefficients_cubic_EOS(l)
        Z_reais = EOS.Z(A, B)
        ph = self.deltaG_molar(EOS, l, ph)
        Z = np.min(Z_reais) * ph + np.max(Z_reais) * (1 - ph)
        Z = Z_reais
        v = Z*R*self.T/self.P #- sum(self.s*EOS.b*l)*6.243864674*10**(-5) #check this unity (ft³/lbmole to m³/mole)
        ksi_phase = (1 / v)
        #ksi_phase = self.P / (Z * kprop.R* self.T)
        Mw_phase = np.sum(l * Mw)
        rho_phase = ksi_phase * np.sum(l * Mw)
        return ksi_phase[0], rho_phase[0]

    '''def TPD(self, z): #ainda não sei onde usar isso
        x = np.zeros(self.Nc)

        #**********************Tangent Plane distance plot*********************#
        t = np.linspace(0.01, 0.99, 0.9 / 0.002) #vetor auxiliar
        TPD = np.zeros(len(t)) ##F

        for i in range(0, len(t)):
            aux = 0;
            lnphiz = self.lnphi(z, 1) #original phase

            #x = np.array([1-t[i],t[i]]) #new phase composition (1-t e t) - apenas válido para Nc=2 acredito eu.
            for k in range(0, kprop.Nc- 1):
                x[k] = (1 - t[i]) / (kprop.Nc- 1)
                x[kprop.Nc- 1] = t[i]

            ''''''O modo que x varia implica no formato de TPD. No presente exemplo,
            a fração molar do segundo componente de x varia direto com t, que é a
            variável de plotagem. Logo, a distancia dos planos tangentes será
            zero em z[Nc-1]. O contrário ocorreria''''''
            lnphix = self.lnphi(x, 0); #new phase (vapor- ph=2)
            for j in range(0,self.Nc):
                fix = math.exp(lnphix[j]) * x[j] * self.P
                fiz = math.exp(lnphiz[j]) * z[j] * self.P
                aux = aux + x[j] * ctes.R* self.T * (math.log(fix / fiz))
                TPD[i] = aux

        plt.figure(0)
        plt.plot(t, TPD)
        plt.xlabel('x')
        plt.ylabel('TPD')
        plt.show()
        return TPD'''
