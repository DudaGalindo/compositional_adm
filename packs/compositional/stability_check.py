"""Check stability of a system and find its composition at a thermodynamic equilibrium."""
import numpy as np
from ..directories import data_loaded
from scipy.misc import derivative
from . import equation_of_state
from ..utils import constants as ctes
import sympy as smp


## Encontrar os pontos estacionarios. Estes correspondem aos pontos nos quais a derivada de g com respeito a Y é 0
## Todas as equações foram confirmadas pelo livro do Dandekar e a biblioteca do thermo
# ph=0 - vapor. ph=1 - liquid.


class StabilityCheck:
    """Check for stability of a thermodynamic equilibrium and returns the
    equilibrium phase compositions (perform the flash calculation)."""

    def __init__(self, fprop):
        self.EOS_class = getattr(equation_of_state, data_loaded['compositional_data']['equation_of_state'])
        self.EOS = self.EOS_class(fprop.T)
        self.ph_L = np.ones(len(fprop.P), dtype = bool)
        self.ph_V = np.zeros(len(fprop.P), dtype = bool)

    def run(self, fprop):
        self.x = fprop.component_molar_fractions[0:ctes.Nc,0,0:len(fprop.P)].T
        self.y = fprop.component_molar_fractions[0:ctes.Nc,1,0:len(fprop.P)].T

        self.equilibrium_ratio_Wilson(fprop)
        ponteiro_flash_all = np.ones(len(fprop.P), dtype = bool)

        ponteiro_flash = np.copy(ponteiro_flash_all)
        dir_flash = np.argwhere(fprop.z.T <= 0)
        ponteiro_flash[dir_flash[:,0]] = False
        if any(ponteiro_flash):

            sp1,sp2 = self.StabilityTest(fprop, np.copy(ponteiro_flash))
            ponteiro_flash[(sp1 > 1) + (sp1 > 1)] = True #os que em TESE deveriam passar para o calculo de flash

        """ Cálculo de flash para todos - sem restrição do teste de estabilidade """
        self.molar_properties(fprop, np.copy(ponteiro_flash_all))

        self.update_EOS_dependent_properties(fprop)
        fprop.component_molar_fractions[0:ctes.Nc,0,:] = self.x.T
        fprop.component_molar_fractions[0:ctes.Nc,1,:] = self.y.T

    def equilibrium_ratio_Wilson(self, fprop):
        self.K = np.exp(5.37 * (1 + ctes.w) * (1 - 1/(fprop.T / ctes.Tc))) / \
                (fprop.P[:,np.newaxis] / ctes.Pc)


    """------------------- Stability test calculation -----------------------"""

    def StabilityTest(self, fprop, ponteiro_stab_check):
        ''' In the lnphi function: 0 stands for vapor phase and 1 for liquid '''

    #****************************INITIAL GUESS******************************#
    ## Both approaches bellow should be used in case the phase is in the critical region

    #*****************************Test one**********************************#
        #Used alone when the phase investigated (z) is clearly vapor like (ph = 0)
        ponteiro = np.copy(ponteiro_stab_check)

        Y = fprop.z.T[ponteiro] / self.K[ponteiro]
        y = Y / np.sum(Y, axis = 1)[:, np.newaxis]
        lnphiz = self.EOS.lnphi(fprop.z.T[ponteiro], fprop.P[ponteiro], self.ph_V[ponteiro])

        while any(ponteiro):
            Y_old = np.copy(Y[ponteiro])
            lnphiy = self.EOS.lnphi(y[ponteiro], fprop.P[ponteiro], self.ph_L[ponteiro])
            Y[ponteiro] = np.exp(np.log(fprop.z.T[ponteiro]) + lnphiz[ponteiro] - lnphiy)
            y[ponteiro] = Y[ponteiro] / np.sum(Y[ponteiro], axis = 1)[:, np.newaxis]
            stop_criteria = np.max(abs(Y[ponteiro] / Y_old - 1), axis = 1)
            ponteiro_aux = ponteiro[ponteiro]
            ponteiro_aux[stop_criteria < 1e-9] = False
            ponteiro[ponteiro] = ponteiro_aux
        stationary_point1 = np.sum(Y, axis = 1)


    #*****************************Test two**********************************#
        #Used alone when the phase investigated (z) is clearly liquid like (ph == 1)
        ponteiro = np.copy(ponteiro_stab_check)

        Y = self.K[ponteiro] * fprop.z.T[ponteiro]
        y = Y / np.sum(Y, axis = 1)[:, np.newaxis]
        lnphiz = self.EOS.lnphi(fprop.z.T[ponteiro], fprop.P[ponteiro], self.ph_L[ponteiro])
        while any(ponteiro):
            Y_old = np.copy(Y[ponteiro])
            lnphiy = self.EOS.lnphi(y[ponteiro], fprop.P[ponteiro], self.ph_V[ponteiro])
            Y[ponteiro] = np.exp(np.log(fprop.z.T[ponteiro]) + lnphiz[ponteiro] - lnphiy)
            y[ponteiro] = Y[ponteiro] / np.sum(Y[ponteiro], axis = 1)[:, np.newaxis]
            stop_criteria = np.max(abs(Y[ponteiro] / Y_old - 1), axis = 1)
            ponteiro_aux = ponteiro[ponteiro]
            ponteiro_aux[stop_criteria < 1e-9] = False
            ponteiro[ponteiro] = ponteiro_aux
        stationary_point2 = np.sum(Y, axis = 1)

        return stationary_point1, stationary_point2


    """-------------------- Biphasic flash calculations ---------------------"""

    def molar_properties(self, fprop, ponteiro):
        if ctes.Nc == 1: ponteiro = self.vapor_pressure(fprop, ponteiro)
        if ctes.Nc == 2: ponteiro = self.molar_properties_Whitson(fprop, ponteiro)
        elif ctes.Nc > 2: ponteiro = self.molar_properties_Yinghui(fprop, ponteiro)


    def deltaG_molar_vectorized(self, l, P, ph):
        lnphi = np.empty([2, len(ph), ctes.Nc])
        lnphi[0,:] = self.EOS.lnphi(l, P, 1 - ph)
        lnphi[1,:] = self.EOS.lnphi(l, P, ph)

        deltaG_molar = np.sum(l * (lnphi[1 - ph,np.arange(len(ph)),:] - lnphi[1*ph, np.arange(len(ph)),:]), axis = 1)
        ph[deltaG_molar<0] = 1 - ph[deltaG_molar<0]
        return ph

    def lnphi_based_on_deltaG(self, l, P, ph):
        ph = self.deltaG_molar_vectorized(l, P, ph)
        return self.EOS.lnphi(l, P, ph)

    def solve_objective_function_Yinghui(self, z1, zi, z, K1, KNc, Ki, K, x, z1_zero):
        x1_min = z1 * (1 - KNc) / (K1 - KNc)
        x1_max = (1 - KNc) / (K1 - KNc)

        vols_z_neg = np.zeros(len(K1), dtype = bool)
        vols_z_neg[np.sum(z < 0, axis = 1, dtype=bool)] = True
        theta = np.ones(z[vols_z_neg].shape)
        KNc_z_neg = KNc[vols_z_neg]
        K1_z_neg = K1[vols_z_neg]
        z1_z_neg = z1[vols_z_neg]
        K_z_neg = K[vols_z_neg]
        z_z_neg = z[vols_z_neg]

        vols_z_neg_num = np.sum(vols_z_neg*1) + 1 - np.sign(np.sum(vols_z_neg*1))
        K_z_neg_K_big1 = K_z_neg[K_z_neg > 1].reshape(vols_z_neg_num,int(len(K_z_neg[K_z_neg>1])/vols_z_neg_num))
        theta = np.ones(z[vols_z_neg].shape)
        theta[K_z_neg > 1] = ((1 - KNc_z_neg[:,np.newaxis]) / (K_z_neg_K_big1 - KNc_z_neg[:,np.newaxis])).ravel()
        aux_eq = (K_z_neg - 1) * z1_z_neg[:, np.newaxis] / (z_z_neg * (K1_z_neg[:, np.newaxis] - 1) /
                theta - (K1_z_neg[:, np.newaxis] - K_z_neg))

        #aux_eq = (K - 1) * z1 / (z * (K1 - 1) / theta - (K1 - K))
        cond = (K_z_neg[z_z_neg != 0] - 1) * z1_z_neg[:,np.newaxis] / z_z_neg[z_z_neg != 0]
        cond_aux = np.ones(cond.shape[0], dtype = bool)
        cond_aux[np.sum(cond <= 0, axis = 1, dtype=bool)] = False
        aux_eq_cond = aux_eq[cond_aux]
        vols_aux = len(cond_aux==True)

        vols_aux = np.sum(cond_aux*1) + 1 - np.sign(np.sum(cond_aux*1))
        aux_eq_cond = aux_eq_cond[aux_eq_cond >= 0].reshape(vols_aux,int(len(aux_eq_cond[aux_eq_cond >= 0])/vols_aux))
        x1_max_aux = np.copy(x1_max[vols_z_neg])
        if len(cond_aux) > 0: x1_max_aux[cond_aux] = np.min(aux_eq_cond, axis = 1)
        x1_max[vols_z_neg] = x1_max_aux

        x1_min_aux = np.copy(x1_min[vols_z_neg])
        x1_min_aux[~cond_aux] = np.max(aux_eq[~cond_aux], axis = 1)
        x1_min_aux[x1_min_aux < 0] = 0
        x1_min[vols_z_neg] = x1_min_aux

        if any(x1_min > x1_max): raise ValueError('There is no physical root')

        x1 = (x1_min + x1_max) / 2

        ponteiro = np.ones(len(x1), dtype = bool)

        while any(ponteiro):
            f = 1 + ((K1[ponteiro] - KNc[ponteiro]) / (KNc[ponteiro] - 1)) * x1[ponteiro] + np.sum(((Ki[ponteiro] - KNc[ponteiro][:, np.newaxis]) /
                (KNc[ponteiro][:, np.newaxis] - 1)) * zi[ponteiro] * (K1[ponteiro][:, np.newaxis] - 1) * x1[ponteiro][:, np.newaxis]
                / ((Ki[ponteiro] - 1) * z1[ponteiro][:, np.newaxis] + (K1[ponteiro][:,np.newaxis] - Ki[ponteiro]) *
                x1[ponteiro][:,np.newaxis]), axis = 1)
            df = ((K1[ponteiro] - KNc[ponteiro]) / (KNc[ponteiro] - 1)) + np.sum(((Ki[ponteiro] - KNc[ponteiro][:, np.newaxis]) /
                (KNc[ponteiro][:, np.newaxis] - 1)) * zi[ponteiro] * z1[ponteiro][:, np.newaxis] * (K1[ponteiro][:, np.newaxis] - 1) *
                (Ki[ponteiro] - 1) / ((Ki[ponteiro] - 1) * z1[ponteiro][:, np.newaxis] + (K1[ponteiro][:, np.newaxis] - Ki[ponteiro]) *
                x1[ponteiro][:, np.newaxis]) ** 2, axis = 1)
            x1[ponteiro] = x1[ponteiro] - f/df #Newton-Raphson iterative method
            x1_aux = x1[ponteiro]
            x1_aux[x1_aux > x1_max] = (x1_min[x1_aux > x1_max] + x1_max[x1_aux > x1_max])/2
            x1_aux[x1_aux < x1_min] = (x1_min[x1_aux < x1_min] + x1_max[x1_aux < x1_min])/2
            x1[ponteiro] = x1_aux
            ponteiro_aux = ponteiro[ponteiro] #o que muda de tamanho
            ponteiro_aux[f < 1e-10] = False
            ponteiro[ponteiro] = ponteiro_aux
            x1_max = x1_max[ponteiro_aux]
            x1_min = x1_min[ponteiro_aux]
            x1_max[f[ponteiro_aux] * df[ponteiro_aux] > 0] = x1[ponteiro][f[ponteiro_aux] * df[ponteiro_aux] > 0]
            x1_min[f[ponteiro_aux] * df[ponteiro_aux] < 0] = x1[ponteiro][f[ponteiro_aux] * df[ponteiro_aux] < 0]


        xi = (K1[:, np.newaxis] - 1) * zi * x1[:, np.newaxis] / ((Ki - 1) * z1[:, np.newaxis] +
            (K1[:, np.newaxis] - Ki) * x1[:, np.newaxis])

        x_not_z1_zero = np.empty(x[~z1_zero].shape)
        x_not_z1_zero[K == K1[:,np.newaxis]] = x1
        x_not_z1_zero[K == KNc[:,np.newaxis]] = 1 - np.sum(xi, axis = 1) - x1
        aux_xi = np.ones(x_not_z1_zero.shape,dtype=bool)
        aux_xi[K == K1[:,np.newaxis]] = False
        aux_xi[K == KNc[:,np.newaxis]] = False
        x_not_z1_zero[aux_xi] = xi.ravel()
        x[~z1_zero] = x_not_z1_zero
        return x


    def Yinghui_method(self, fprop, ponteiro):

        """ Shaping K to Nc-2 components by removing K1 and KNc and z to Nc-2
        components by removing z1 and zNc """
        K = self.K[ponteiro]
        x = self.x[ponteiro]
        z = fprop.z.T[ponteiro]
        K1 = np.max(K, axis=1); KNc = np.min(K, axis=1)
        z1 = z[K == K1[:,np.newaxis]]

        aux = np.ones(K.shape, dtype = bool)
        aux[K == K1[:,np.newaxis]] = False
        aux[K == KNc[:,np.newaxis]] = False
        Ki = K[aux]
        zi = z[aux]

        ''' Reshaping them into the original matricial form '''
        vols_ponteiro = np.sum(ponteiro*1) + 1 - np.sign(np.sum(ponteiro*1))
        Ki = Ki.reshape(vols_ponteiro, int(len(Ki)/vols_ponteiro))
        zi = zi.reshape(vols_ponteiro, int(len(zi)/vols_ponteiro))

        #starting x

        """ Solution """
        z1_zero = np.zeros(vols_ponteiro,dtype=bool)
        z1_zero[z1 == 0] = True

        x = self.solve_objective_function_Yinghui(z1[~z1_zero], zi[~z1_zero], z[~z1_zero],
                                K1[~z1_zero], KNc[~z1_zero], Ki[~z1_zero], K[~z1_zero], x, z1_zero)

        '''Explicit Calculation of xi'''
        #self.solve_objective_function_Yinghui_explicitly()
        z_z1_zero = z[z1_zero]
        K_z1_zero = K[z1_zero]
        K_KNc_z1_zero = K_z1_zero[K_z1_zero == KNc[z1_zero][:,np.newaxis]]

        aux_xNc = np.zeros(K_z1_zero.shape, dtype = bool); aux_x1 = np.copy(aux_xNc)
        aux_xNc[K_z1_zero == KNc[z1_zero][:,np.newaxis]] = True
        aux_x1[K_z1_zero == K1[z1_zero][:,np.newaxis]] = True
        aux_xi = aux_xNc + aux_x1
        aux_xi = ~aux_xi
        xi_z1_zero = (K1[z1_zero][:, np.newaxis] - 1) * zi[z1_zero] / (K1[z1_zero][:, np.newaxis] - Ki[z1_zero])
        x_z1_zero = np.zeros(x[z1_zero].shape)
        x_z1_zero[aux_xNc] = (K1[z1_zero] - 1) * z_z1_zero[aux_xNc] / (K1[z1_zero] - K_z1_zero[aux_xNc])
        x_z1_zero[aux_x1] = 1 - np.sum(xi_z1_zero, axis = 1) - np.sum(x_z1_zero, axis = 1)
        x_z1_zero[aux_xi] = xi_z1_zero.ravel()
        x[z1_zero] = x_z1_zero
        self.x[ponteiro] = x
        self.y[ponteiro] = selfpropf.K[ponteiro] * self.x[ponteiro]

    def molar_properties_Yinghui(self, fprop, ponteiro):
        #razao = fl/fv -> an arbitrary vector to enter in the iterative mode

        razao = np.ones(z[ponteiro].shape)/2
        ponteiro_save = np.copy(ponteiro)
        while any(ponteiro):
            self.Yinghui_method(fprop, ponteiro)
            lnphil = self.lnphi_based_on_deltaG(self.x[ponteiro], fprop.P[ponteiro], self.ph_L[ponteiro])
            lnphiv = self.lnphi_based_on_deltaG(self.y[ponteiro], fprop.P[ponteiro], self.ph_V[ponteiro])
            fl = np.exp(lnphil) * (self.x[ponteiro] * fprop.P[ponteiro][:, np.newaxis])
            fv = np.exp(lnphiv) * (self.y[ponteiro] * fprop.P[ponteiro][:, np.newaxis])
            razao[ponteiro] = np.divide(fl, fv, out = razao[ponteiro] / razao[ponteiro] * (1 + 1e-10),
                              where = fv != 0)
            self.K[ponteiro] = razao[ponteiro] * self.K[ponteiro]
            stop_criteria = np.max(abs(razao[ponteiro] - 1), axis = 1)
            ponteiro_aux = ponteiro[ponteiro]
            ponteiro_aux[stop_criteria < 1e-9] = False
            ponteiro[ponteiro] = ponteiro_aux

        V = (fprop.z.T[ponteiro_save][self.x[ponteiro_save] != 0] - self.x[ponteiro_save][self.x[ponteiro_save] != 0]) / \
                          (self.y[ponteiro_save][self.x[ponteiro_save] != 0] - self.x[ponteiro_save][self.x[ponteiro_save] != 0])
        #vols_V = np.sum(self.x[ponteiro_save] == 0, dtype=bool, axis = 1)
        vv = np.argwhere((self.x[ponteiro_save]!=0) == True)
        vols_V, ind = np.unique(vv[:,0],return_index = True)
        fprop.V[ponteiro_save] = V[ind]
        fprop.L[ponteiro_save] = 1. - fprop.V[ponteiro_save]
        return ponteiro_save

    def solve_objective_function_Whitson_for_V(self, fprop, V, Vmax, Vmin, ponteiro):

        ponteiro_save = np.copy(ponteiro)
        Vold = np.copy(V)
        while any(ponteiro):
            Vold[ponteiro] = np.copy(V[ponteiro])
            f = np.sum((self.K[ponteiro] - 1) * fprop.z.T[ponteiro] / (1 + V[ponteiro][:,np.newaxis] *
                (self.K[ponteiro] - 1)), axis = 1)
            df = - np.sum((self.K[ponteiro] - 1) ** 2 * fprop.z.T[ponteiro] / (1 + V[ponteiro][:,np.newaxis] *
                (self.K[ponteiro] - 1)) ** 2, axis = 1)
            V[ponteiro] = V[ponteiro] - f / df #Newton-Raphson iterative method
            V_aux = np.copy(V[ponteiro])
            V_aux[V_aux > Vmax[ponteiro]] = Vmax[ponteiro][V_aux > Vmax[ponteiro]] #+ Vold[ponteiro][V_aux > Vmax[ponteiro]]) * 0.5 #(Vmax + Vold)/2
            V_aux[V_aux < Vmin[ponteiro]] = Vmin[ponteiro][V_aux < Vmin[ponteiro]] #+ Vold[ponteiro][V_aux < Vmin[ponteiro]]) * 0.5 #(Vmax + Vold)/2
            V[ponteiro] = np.copy(V_aux)
            stop_criteria = abs(V[ponteiro] / Vold[ponteiro] - 1)
            ponteiro_aux = ponteiro[ponteiro]
            ponteiro_aux[stop_criteria < 1e-9] = False
            ponteiro[ponteiro] = ponteiro_aux

        fprop.V[ponteiro_save] = V[ponteiro_save] + 1e-15 #manipulação - precisão das contas é >1e-15 infelizmente n consigo mudar agora
        self.x[ponteiro_save] = fprop.z.T[ponteiro_save] / (1 + fprop.V[ponteiro_save][:,np.newaxis] * (self.K[ponteiro_save] - 1))
        self.y[ponteiro_save] = self.K[ponteiro_save] * self.x[ponteiro_save]
        fprop.L[ponteiro_save] = 1. - fprop.V[ponteiro_save]


    def molar_properties_Whitson(self, fprop, ponteiro):
        Lmax = np.max(self.K, axis = 1)/(np.max(self.K, axis = 1) - 1)
        Lmin = np.min(self.K, axis = 1)/(np.min(self.K, axis = 1) - 1)
        Vmax = 1. - Lmin
        Vmin = 1. - Lmax
        #Vmin = ((K1-KNc)*z[self.K==K1]-(1-KNc))/((1-KNc)*(K1-1))
        #proposed by Li et al for Whitson method
        V = (Vmin + Vmax) * 0.5 + 1e-15 #manipulação - precisão das contas é >1e-15 infelizmente n consigo mudar agora

        ponteiro[np.sum((1 + V[ponteiro][:,np.newaxis] * (self.K[ponteiro] - 1)==0),axis=1,dtype = bool)] = False
        import pdb; pdb.set_trace()
        ponteiro_save = np.copy(ponteiro)
        razao = np.ones(fprop.z.T.shape)/2
        while any(ponteiro):
            self.solve_objective_function_Whitson_for_V(fprop, V, Vmax, Vmin, np.copy(ponteiro))
            lnphil = self.lnphi_based_on_deltaG(self.x[ponteiro], fprop.P[ponteiro], self.ph_L[ponteiro])
            lnphiv = self.lnphi_based_on_deltaG(self.y[ponteiro], fprop.P[ponteiro], self.ph_V[ponteiro])
            fv = np.exp(lnphiv) * (self.y[ponteiro] * fprop.P[ponteiro][:,np.newaxis])
            fl = np.exp(lnphil) * (self.x[ponteiro] * fprop.P[ponteiro][:,np.newaxis])
            razao[ponteiro] = np.divide(fl, fv, out = razao[ponteiro] / razao[ponteiro] * (1 + 1e-10),
                              where = fv != 0)
            self.K[ponteiro] = razao[ponteiro] * self.K[ponteiro]
            stop_criteria = np.max(abs(fv / fl - 1), axis = 1)
            ponteiro_aux = ponteiro[ponteiro]
            ponteiro_aux[stop_criteria < 1e-9] = False
            ponteiro[ponteiro] = ponteiro_aux
        return ponteiro_save

    def vapor_pressure(self, fprop, ponteiro):
        #Isso vem de uma junção da Lei de Dalton com a Lei de Raoult
        ponteiro_save = np.copy(ponteiro)
        self.Pv = np.empty(len(fprop.P))
        self.Pv[ponteiro] = (self.K[ponteiro] * fprop.P[ponteiro][:,np.newaxis]).ravel()
        self.x = fprop.z.T; self.y = fprop.z.T
        razao = np.ones(fprop.z.T.shape)/2
        while any(ponteiro):
            lnphil = self.lnphi_based_on_deltaG(self.x[ponteiro], self.Pv[ponteiro], self.ph_L[ponteiro])
            lnphiv = self.lnphi_based_on_deltaG(self.y[ponteiro], self.Pv[ponteiro], self.ph_V[ponteiro])
            fv = np.exp(lnphiv) * (self.y[ponteiro] * self.Pv[ponteiro][:,np.newaxis])
            fl = np.exp(lnphil) * (self.x[ponteiro] * self.Pv[ponteiro][:,np.newaxis])
            razao[ponteiro] = np.divide(fl, fv, out = razao[ponteiro] / razao[ponteiro] * (1 + 1e-10),
                              where = fv != 0)
            stop_criteria = np.max(abs(fv / fl - 1), axis = 1)
            self.Pv[ponteiro] = (razao[ponteiro] * self.Pv[ponteiro][:, np.newaxis]).ravel()
            ponteiro_aux = ponteiro[ponteiro]
            ponteiro_aux[stop_criteria < 1e-9] = False
            ponteiro[ponteiro] = ponteiro_aux

        L = fprop.L[ponteiro_save]
        V = fprop.V[ponteiro_save]
        L[fprop.P[ponteiro_save] > self.Pv[ponteiro_save]] = 1
        V[fprop.P[ponteiro_save] > self.Pv[ponteiro_save]] = 0.
        L[fprop.P[ponteiro_save] < self.Pv[ponteiro_save]] = 0.
        V[fprop.P[ponteiro_save] < self.Pv[ponteiro_save]] = 1
        fprop.L[ponteiro_save] = L
        fprop.V[ponteiro_save] = V

        return ponteiro_save

        #Pv = self.K[ponteiro] * fprop.P[ponteiro][:,np.newaxis]
        #self.Pv = np.sum(self.x[ponteiro] * Pv[ponteiro], axis = 1)

    def update_EOS_dependent_properties(self, fprop):
        #self.EOS = self.EOS_class(self.P, fprop.T)

        fprop.phase_molar_densities[0,0,:], fprop.phase_densities[0,0,:] = self.get_EOS_dependent_properties(fprop.T, self.x, fprop.P, self.ph_L)
        fprop.phase_molar_densities[0,1,:], fprop.phase_densities[0,1,:] = self.get_EOS_dependent_properties(fprop.T, self.y, fprop.P, self.ph_V)

    def get_EOS_dependent_properties(self, T, l, P, ph):
        #l - any phase molar composition
        # This won't work for everything, I'm sure of that. Still need, tho, to check the cases that this won't work.
        #l = l[:,np.newaxis]
        #self.EOS_class(fprop.T)
        A, B = self.EOS_class(T).coefficients_cubic_EOS_vectorized(l.T, P)
        ph = self.deltaG_molar_vectorized(l, P, ph)
        Z_func = np.vectorize(self.EOS_class.Z)
        Z = Z_func(A, B, ph)
        v = Z * ctes.R * T / P
        ksi_phase = 1 / v
        #ksi_phase = self.P / (Z * ctes.R* fprop.T)
        Mw_phase = np.sum(l * ctes.Mw, axis = 1)
        rho_phase = ksi_phase * np.sum(l * ctes.Mw, axis = 1)
        return ksi_phase, rho_phase


    '''def TPD(self, z): #ainda não sei onde usar isso
        x = np.zeros(self.Nc)

        #**********************Tangent Plane distance plot*********************#
        t = np.linspace(0.01, 0.99, 0.9 / 0.002) #vetor auxiliar
        TPD = np.zeros(len(t)) ##F

        for i in range(0, len(t)):
            aux = 0;
            lnphiz = self.lnphi(z, 1) #original phase

            #x = np.array([1-t[i],t[i]]) #new phase composition (1-t e t) - apenas válido para Nc=2 acredito eu.
            for k in range(0, ctes.Nc- 1):
                x[k] = (1 - t[i]) / (ctes.Nc- 1)
                x[ctes.Nc- 1] = t[i]

            ''''''O modo que x varia implica no formato de TPD. No presente exemplo,
            a fração molar do segundo componente de x varia direto com t, que é a
            variável de plotagem. Logo, a distancia dos planos tangentes será
            zero em z[Nc-1]. O contrário ocorreria''''''
            lnphix = self.lnphi(x, 0); #new phase (vapor- ph=2)
            for j in range(0,self.Nc):
                fix = math.exp(lnphix[j]) * x[j] * self.P
                fiz = math.exp(lnphiz[j]) * z[j] * self.P
                aux = aux + x[j] * ctes.R* fprop.T * (math.log(fix / fiz))
                TPD[i] = aux

        plt.figure(0)
        plt.plot(t, TPD)
        plt.xlabel('x')
        plt.ylabel('TPD')
        plt.show()
        return TPD'''
