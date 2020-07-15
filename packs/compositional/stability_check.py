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
        self.EOS = ctes.EOS_class(fprop.T) #só para os casos isotérmicos - non isothermal entrariam em run (eu acho)
        self.ph_L = np.ones(len(fprop.P), dtype = bool)
        self.ph_V = np.zeros(len(fprop.P), dtype = bool)
        self.Pv = np.array(data_loaded['compositional_data']['component_data']['Pv'])
        self.Pv[fprop.T < ctes.Tc] = self.vapor_pressure_pure_substancies(fprop)


    def run(self, fprop):
        self.x = fprop.component_molar_fractions[0:ctes.Nc, 0, 0:len(fprop.P)]
        self.y = fprop.component_molar_fractions[0:ctes.Nc, 1, 0:len(fprop.P)]
        self.equilibrium_ratio_Wilson(fprop)

        ponteiro_flash = np.zeros(len(fprop.P), dtype = bool)
        dir_flash = np.argwhere(fprop.z.T <= 0)
        ponteiro_flash[dir_flash[:,0]] = True

        sp1,sp2 = self.StabilityTest(fprop, np.copy(~ponteiro_flash))

        ponteiro_aux = ponteiro_flash[~ponteiro_flash]
        ponteiro_aux[(np.round(sp1,13) > 1) + (np.round(sp2,13) > 1)] = True #os que devem pass=r para o calculo de flash
        ponteiro_flash[~ponteiro_flash] = ponteiro_aux
        #if len(fprop.P) > 1: ponteiro_flash[19] = False
        self.molar_properties(fprop, np.copy(ponteiro_flash))
        '''if len(fprop.P) > 1:
            ponteiro_flash[19] = True
            self.x[:,19] = fprop.z[:,19]
            self.y[:,19] = fprop.z[:,19]
            fprop.L[19] = 0.
            fprop.V[19] = 1.

        self.x[:,fprop.L>1] = fprop.z[:,fprop.L>1]
        self.y[:,fprop.L>1] = fprop.z[:,fprop.L>1]
        self.x[:,fprop.L<0] = fprop.z[:,fprop.L<0]
        self.y[:,fprop.L<0] = fprop.z[:,fprop.L<0]

        ponteiro_flash[fprop.L>1] = False
        ponteiro_flash[fprop.L<0] = False'''
        self.bubble_point_pressure(fprop, ~ponteiro_flash)
        self.update_EOS_dependent_properties(fprop)
        fprop.component_molar_fractions[0:ctes.Nc, 0, :] = self.x
        fprop.component_molar_fractions[0:ctes.Nc, 1, :] = self.y
        #fprop.L[fprop.L>1] = 1
        #fprop.V[fprop.V<0] = 0

    def vapor_pressure_pure_substancies(self, fprop):
        '''Lee-Kesler Correlation - only valid for T < Tc'''
        Tr = fprop.T/ctes.Tc

        A = 5.92714 - 6.09648 / Tr - 1.2886 * np.log(Tr) + 0.16934 * Tr**6
        B = 15.2518 - 15.6875 / Tr - 13.4721 * np.log(Tr) + 0.4357 * Tr**6
        Pv = ctes.Pc * np.exp(A + ctes.w * B)
        return Pv[fprop.T < ctes.Tc]

    def equilibrium_ratio_Wilson(self, fprop):
        self.K = np.exp(5.37 * (1 + ctes.w) * (1 - 1/(fprop.T / ctes.Tc)))[:,np.newaxis] / \
                (fprop.P / ctes.Pc[:,np.newaxis])

    """------------------- Stability test calculation -----------------------"""

    def StabilityTest(self, fprop, ponteiro_stab_check):
        ''' In the lnphi function: 0 stands for vapor phase and 1 for liquid '''

    #****************************INITIAL GUESS******************************#
    ## Both approaches bellow should be used in case the phase is in the critical region

    #*****************************Test one**********************************#
        #Used alone when the phase investigated (z) is clearly vapor like (ph = 0)

        ponteiro = np.copy(ponteiro_stab_check)
        Y = np.empty(fprop.z.shape)
        lnphiz = np.empty(fprop.z.shape)

        Y[:,ponteiro] = fprop.z[:,ponteiro] / self.K[:,ponteiro]
        y = Y / np.sum(Y, axis = 0)[np.newaxis,:]
        lnphiz[:,ponteiro] = self.EOS.lnphi(fprop.z[:,ponteiro], fprop.P[ponteiro], self.ph_V[ponteiro])
        while any(ponteiro):
            Y_old = np.copy(Y[:,ponteiro])
            lnphiy = self.EOS.lnphi(y[:,ponteiro], fprop.P[ponteiro], self.ph_L[ponteiro])
            Y[:,ponteiro] = np.exp(np.log(fprop.z[:,ponteiro]) + lnphiz[:,ponteiro] - lnphiy)
            y[:,ponteiro] = Y[:,ponteiro] / np.sum(Y[:,ponteiro], axis = 0)[np.newaxis,:]
            stop_criteria = np.max(abs(Y[:,ponteiro] / Y_old - 1), axis = 0)
            ponteiro_aux = ponteiro[ponteiro]
            ponteiro_aux[stop_criteria < 1e-9] = False
            ponteiro[ponteiro] = ponteiro_aux
        stationary_point1 = np.sum(Y, axis = 0)


    #*****************************Test two**********************************#
        #Used alone when the phase investigated (z) is clearly liquid like (ph == 1)
        ponteiro = np.copy(ponteiro_stab_check)

        Y[:,ponteiro] = self.K[:,ponteiro] * fprop.z[:,ponteiro]
        y = Y / np.sum(Y, axis = 0)[np.newaxis,:]
        lnphiz[:,ponteiro] = self.EOS.lnphi(fprop.z[:,ponteiro], fprop.P[ponteiro], self.ph_L[ponteiro])
        while any(ponteiro):
            Y_old = np.copy(Y[:,ponteiro])
            lnphiy = self.EOS.lnphi(y[:,ponteiro], fprop.P[ponteiro], self.ph_V[ponteiro])
            Y[:,ponteiro] = np.exp(np.log(fprop.z[:,ponteiro]) + lnphiz[:,ponteiro] - lnphiy)
            y[:,ponteiro] = Y[:,ponteiro] / np.sum(Y[:,ponteiro], axis = 0)[np.newaxis,:]
            stop_criteria = np.max(abs(Y[:,ponteiro] / Y_old - 1), axis = 0)
            ponteiro_aux = ponteiro[ponteiro]
            ponteiro_aux[stop_criteria < 1e-9] = False
            ponteiro[ponteiro] = ponteiro_aux
        stationary_point2 = np.sum(Y, axis = 0)

        '''Analogia feita com base em observações - pode estar errado ainda.
        Isto "substitui" o cálculo da pressão do ponto de bolha'''
        L = fprop.L[ponteiro_stab_check]
        L[(np.round(stationary_point1,13)==1) * (np.round(stationary_point2,13)<=1)] = 1.
        L[(np.round(stationary_point1,13)<1) * (np.round(stationary_point2,13)==1)] = 0.
        fprop.L[ponteiro_stab_check] = L
        fprop.V[ponteiro_stab_check] = 1 - fprop.L[ponteiro_stab_check]
        self.x[:,ponteiro_stab_check] = fprop.z[:,ponteiro_stab_check]
        self.y[:,ponteiro_stab_check] = fprop.z[:,ponteiro_stab_check]
        return stationary_point1, stationary_point2


    """-------------------- Biphasic flash calculations ---------------------"""

    def molar_properties(self, fprop, ponteiro):
        if ctes.Nc <= 5: self.molar_properties_Whitson(fprop, ponteiro)
        else: self.molar_properties_Yinghui(fprop, ponteiro)

    def deltaG_molar_vectorized(self, l, P, ph):
        lnphi = np.empty([2, ctes.Nc, len(ph)])
        lnphi[0,:] = self.EOS.lnphi(l, P, 1 - ph)
        lnphi[1,:] = self.EOS.lnphi(l, P, ph)

        deltaG_molar = np.sum(l * (lnphi[1 - ph, : ,np.arange(len(ph))] - lnphi[1*ph, :, np.arange(len(ph))]).T, axis = 0)
        ph[deltaG_molar<0] = 1 - ph[deltaG_molar<0]
        return ph

    def lnphi_based_on_deltaG(self, l, P, ph):
        ph = self.deltaG_molar_vectorized(l, P, ph)
        return self.EOS.lnphi(l, P, ph)

    def solve_objective_function_Yinghui(self, z1, zi, K1, KNc, Ki, K, x):
        x1_min = z1 * (1 - KNc) / (K1 - KNc)
        x1_max = (1 - KNc) / (K1 - KNc)
        vols_zi_neg = np.zeros(len(K1), dtype = bool)
        vols_zi_neg[np.sum(zi < 0, axis = 0, dtype=bool)] = True

        KNc_z_neg = KNc[vols_zi_neg]
        K1_z_neg = K1[vols_zi_neg]
        z1_z_neg = z1[vols_zi_neg]
        zi_z_neg = zi[:,vols_zi_neg]
        Ki_z_neg = Ki[:,vols_zi_neg]

        vols_zi_neg_num = np.sum(vols_zi_neg*1) + 1 - np.sign(np.sum(vols_zi_neg*1))
        Ki_z_neg_K_big1 = Ki_z_neg[Ki_z_neg > 1].reshape(int(len(Ki_z_neg[Ki_z_neg>1])/vols_zi_neg_num),vols_zi_neg_num)
        theta = np.ones(zi[:,vols_zi_neg].shape)

        theta[Ki_z_neg > 1] = ((1 - KNc_z_neg[np.newaxis,:]) / (Ki_z_neg_K_big1 - KNc_z_neg[np.newaxis,:])).ravel()
        aux_eq = (Ki_z_neg - 1) * z1_z_neg[np.newaxis,:] / (zi_z_neg * (K1_z_neg[np.newaxis,:] - 1) /
                theta - (K1_z_neg[np.newaxis,:] - Ki_z_neg))

        #aux_eq = (K - 1) * z1 / (z * (K1 - 1) / theta - (K1 - K))
        cond = (Ki_z_neg[zi_z_neg != 0] - 1) * z1_z_neg[np.newaxis,:] / zi_z_neg[zi_z_neg != 0]
        cond_aux = np.ones(cond.shape[1], dtype = bool)
        cond_aux[np.sum(cond <= 0, axis = 0, dtype=bool)] = False
        aux_eq_cond = aux_eq[cond_aux]
        vols_aux = len(cond_aux==True)

        vols_aux = np.sum(cond_aux*1) + 1 - np.sign(np.sum(cond_aux*1))
        aux_eq_cond = aux_eq_cond[aux_eq_cond >= 0].reshape(vols_aux,int(len(aux_eq_cond[aux_eq_cond >= 0])/vols_aux))
        x1_max_aux = np.copy(x1_max[vols_zi_neg])
        if any(cond_aux): x1_max_aux[cond_aux] = np.min(aux_eq_cond, axis = 0)
        x1_max[vols_zi_neg] = x1_max_aux

        x1_min_aux = np.copy(x1_min[vols_zi_neg])
        if any(~cond_aux): x1_min_aux[~cond_aux] = np.max(aux_eq[~cond_aux], axis = 0)
        x1_min_aux[x1_min_aux < 0] = 0
        x1_min[vols_zi_neg] = x1_min_aux

        if any(x1_min > x1_max):
            import pdb; pdb.set_trace()
            raise ValueError('There is no physical root')

        x1 = (x1_min + x1_max) / 2

        ponteiro = np.ones(len(x1), dtype = bool)

        while any(ponteiro):
            f = 1 + ((K1[ponteiro] - KNc[ponteiro]) / (KNc[ponteiro] - 1)) * x1[ponteiro] + np.sum(((Ki[:,ponteiro] - KNc[ponteiro][np.newaxis,:]) /
                (KNc[ponteiro][np.newaxis,:] - 1)) * zi[:,ponteiro] * (K1[ponteiro][np.newaxis,:] - 1) * x1[ponteiro][np.newaxis,:]
                / ((Ki[:,ponteiro] - 1) * z1[ponteiro][np.newaxis,:] + (K1[ponteiro][np.newaxis,:] - Ki[:,ponteiro]) *
                x1[ponteiro][np.newaxis,:]), axis = 0)
            df = ((K1[ponteiro] - KNc[ponteiro]) / (KNc[ponteiro] - 1)) + np.sum(((Ki[:,ponteiro] - KNc[ponteiro][np.newaxis,:]) /
                (KNc[ponteiro][np.newaxis,:] - 1)) * zi[:,ponteiro] * z1[ponteiro][np.newaxis,:] * (K1[ponteiro][np.newaxis,:] - 1) *
                (Ki[:,ponteiro] - 1) / ((Ki[:,ponteiro] - 1) * z1[ponteiro][np.newaxis,:] + (K1[ponteiro][np.newaxis,:] - Ki[:,ponteiro]) *
                x1[ponteiro][np.newaxis,:]) ** 2, axis = 0)
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


        xi = (K1[np.newaxis,:] - 1) * zi * x1[np.newaxis,:] / ((Ki - 1) * z1[np.newaxis,:] +
            (K1[np.newaxis,:] - Ki) * x1[np.newaxis,:])

        x_not_z1_zero = np.copy(x)
        x_not_z1_zero[K == K1[np.newaxis,:]] = x1
        x_not_z1_zero[K == KNc[np.newaxis,:]] = 1 - np.sum(xi, axis = 0) - x1
        aux_xi = np.ones(x_not_z1_zero.shape,dtype=bool)
        aux_xi[K == K1[np.newaxis,:]] = False
        aux_xi[K == KNc[np.newaxis,:]] = False
        x_not_z1_zero[aux_xi] = xi.ravel()
        return x_not_z1_zero

    def Yinghui_method(self, fprop, ponteiro):

        """ Shaping K to Nc-2 components by removing K1 and KNc and z to Nc-2
        components by removing z1 and zNc """
        K = self.K[:,ponteiro]
        x = self.x[:,ponteiro]
        z = fprop.z[:,ponteiro]
        K1 = np.max(K, axis = 0); KNc = np.min(K, axis = 0)
        z1 = z[K == K1[np.newaxis,:]]

        aux = np.ones(K.shape, dtype = bool)
        aux[K == K1[np.newaxis,:]] = False
        aux[K == KNc[np.newaxis,:]] = False
        Ki = K[aux]
        zi = z[aux]

        ''' Reshaping them into the original matricial form '''
        vols_ponteiro = np.sum(ponteiro*1) + 1 - np.sign(np.sum(ponteiro*1))
        Ki = Ki.reshape(int(len(Ki)/vols_ponteiro), vols_ponteiro)
        zi = zi.reshape(int(len(zi)/vols_ponteiro), vols_ponteiro)

        #starting x

        """ Solution """
        x[:,~(z1 == 0)] = self.solve_objective_function_Yinghui(z1[~(z1 == 0)], zi[:,~(z1 == 0)],
                                                K1[~(z1 == 0)], KNc[~(z1 == 0)], Ki[:,~(z1 == 0)],
                                                K[:,~(z1 == 0)], x[:,~(z1 == 0)])

        '''Explicit Calculation of xi'''
        #self.solve_objective_function_Yinghui_explicitly()
        z_z1_zero = z[:,z1==0]
        K_z1_zero = K[:,z1==0]
        K_KNc_z1_zero = K_z1_zero[K_z1_zero == KNc[z1==0][np.newaxis,:]]

        aux_xNc = np.zeros(K_z1_zero.shape, dtype = bool); aux_x1 = np.copy(aux_xNc)
        aux_xNc[K_z1_zero == KNc[z1==0][np.newaxis,:]] = True
        aux_x1[K_z1_zero == K1[z1==0][np.newaxis,:]] = True
        aux_xi = ~(aux_xNc + aux_x1)
        xi_z1_zero = ((K1[z1 == 0][np.newaxis,:] - 1) * zi[:,z1 == 0] / (K1[z1 == 0][np.newaxis,:] - Ki[:,z1 == 0]))
        x_z1_zero = np.zeros(x[:,z1 == 0].shape)
        x_z1_zero[aux_xNc] = (K1[z1 == 0] - 1) * z_z1_zero[aux_xNc] / (K1[z1 == 0] - K_z1_zero[aux_xNc])
        x_z1_zero[aux_xi] = xi_z1_zero.ravel()
        x_z1_zero[aux_x1] = 1 - np.sum(x_z1_zero, axis = 0)
        x[:,z1 == 0] = x_z1_zero
        self.x[:,ponteiro] = x
        self.y[:,ponteiro] = self.K[:,ponteiro] * self.x[:,ponteiro]

    def molar_properties_Yinghui(self, fprop, ponteiro):
        #razao = fl/fv -> an arbitrary vector to enter in the iterative mode

        razao = np.ones(fprop.z.shape)/2
        ponteiro_save = np.copy(ponteiro)
        while any(ponteiro):
            self.Yinghui_method(fprop, ponteiro)
            lnphil = self.lnphi_based_on_deltaG(self.x[:,ponteiro], fprop.P[ponteiro], self.ph_L[ponteiro])
            lnphiv = self.lnphi_based_on_deltaG(self.y[:,ponteiro], fprop.P[ponteiro], self.ph_V[ponteiro])
            fl = np.exp(lnphil) * (self.x[:,ponteiro] * fprop.P[ponteiro][np.newaxis,:])
            fv = np.exp(lnphiv) * (self.y[:,ponteiro] * fprop.P[ponteiro][np.newaxis,:])
            razao[:,ponteiro] = np.divide(fl, fv, out = razao[:,ponteiro] / razao[:,ponteiro] * (1 + 1e-10),
                              where = fv != 0)
            self.K[:,ponteiro] = razao[:,ponteiro] * self.K[:,ponteiro]
            stop_criteria = np.max(abs(fv / fl - 1), axis = 0)
            ponteiro_aux = ponteiro[ponteiro]
            ponteiro_aux[stop_criteria < 1e-9] = False
            ponteiro[ponteiro] = ponteiro_aux

        V = (fprop.z[:,ponteiro_save][self.x[:,ponteiro_save] != 0] - self.x[:,ponteiro_save][self.x[:,ponteiro_save] != 0]) / \
                          (self.y[:,ponteiro_save][self.x[:,ponteiro_save] != 0] - self.x[:,ponteiro_save][self.x[:,ponteiro_save] != 0])
        #vols_V = np.sum(self.x[:,ponteiro_save] == 0, dtype=bool, axis = 0)
        vv = np.argwhere((self.x[:,ponteiro_save]!=0) == True)
        vols_V, ind = np.unique(vv[:,1],return_index = True)
        fprop.V[ponteiro_save] = V[ind]
        fprop.L[ponteiro_save] = 1. - fprop.V[ponteiro_save]

    def solve_objective_function_Whitson_for_V(self, fprop, V, Vmax, Vmin, ponteiro):

        ponteiro_save = np.copy(ponteiro)
        Vold = np.copy(V)
        while any(ponteiro):
            Vold[ponteiro] = np.copy(V[ponteiro])
            f = np.sum((self.K[:,ponteiro] - 1) * fprop.z[:,ponteiro] / (1 + V[ponteiro][np.newaxis,:] *
                (self.K[:,ponteiro] - 1)), axis = 0)
            df = - np.sum((self.K[:,ponteiro] - 1) ** 2 * fprop.z[:,ponteiro] / (1 + V[ponteiro][np.newaxis,:] *
                (self.K[:,ponteiro] - 1)) ** 2, axis = 0)
            V[ponteiro] = V[ponteiro] - f / df #Newton-Raphson iterative method
            V_aux = V[ponteiro]
            V_aux[V_aux > Vmax[ponteiro]] = Vmax[ponteiro][V_aux > Vmax[ponteiro]] #+ Vold[ponteiro][V_aux > Vmax[:,ponteiro]]) * 0.5 #(Vmax + Vold)/2
            V_aux[V_aux < Vmin[ponteiro]] = Vmin[ponteiro][V_aux < Vmin[ponteiro]] #+ Vold[ponteiro][V_aux < Vmin[ponteiro]]) * 0.5 #(Vmax + Vold)/2
            V[ponteiro] = V_aux
            stop_criteria = abs(V[ponteiro] / Vold[ponteiro] - 1)
            ponteiro_aux = ponteiro[ponteiro]
            ponteiro_aux[stop_criteria < 1e-9] = False
            ponteiro[ponteiro] = ponteiro_aux

        fprop.V[ponteiro_save] = V[ponteiro_save]
        self.x[:,ponteiro_save] = fprop.z[:,ponteiro_save] / (1 + fprop.V[ponteiro_save][np.newaxis,:] *
                                (self.K[:,ponteiro_save] - 1))
        self.y[:,ponteiro_save] = self.K[:,ponteiro_save] * self.x[:,ponteiro_save]
        fprop.L[ponteiro_save] = 1. - fprop.V[ponteiro_save]


    def molar_properties_Whitson(self, fprop, ponteiro):
        Lmax = np.max(self.K, axis = 0)/(np.max(self.K, axis = 0) - 1)
        Lmin = np.min(self.K, axis = 0)/(np.min(self.K, axis = 0) - 1)
        Vmax = 1. - Lmin
        Vmin = 1. - Lmax
        #Vmin = ((K1-KNc)*z[self.K==K1]-(1-KNc))/((1-KNc)*(K1-1))
        #proposed by Li et al for Whitson method
        V = fprop.V
        V[ponteiro] = (Vmin[ponteiro] + Vmax[ponteiro]) * 0.5
        ponteiro_save = np.copy(ponteiro)
        razao = np.ones(fprop.z.shape)/2
        while any(ponteiro):
            self.solve_objective_function_Whitson_for_V(fprop, V, Vmax, Vmin, np.copy(ponteiro))
            lnphil = self.lnphi_based_on_deltaG(self.x[:,ponteiro], fprop.P[ponteiro], self.ph_L[ponteiro])
            lnphiv = self.lnphi_based_on_deltaG(self.y[:,ponteiro], fprop.P[ponteiro], self.ph_V[ponteiro])
            fv = np.exp(lnphiv) * (self.y[:,ponteiro] * fprop.P[ponteiro][np.newaxis,:])
            fl = np.exp(lnphil) * (self.x[:,ponteiro] * fprop.P[ponteiro][np.newaxis,:])
            razao[:,ponteiro] = np.divide(fl, fv, out = razao[:,ponteiro] / razao[:,ponteiro] * (1 + 1e-10),
                              where = fv != 0)
            self.K[:,ponteiro] = razao[:,ponteiro] * self.K[:,ponteiro]
            stop_criteria = np.max(abs(fv / fl - 1), axis = 0)
            ponteiro_aux = ponteiro[ponteiro]
            ponteiro_aux[stop_criteria < 1e-9] = False
            ponteiro[ponteiro] = ponteiro_aux


    def get_dlnphidP(self, T, xij, P, ph):
        #self.EOS = ctes.EOS_class(T)
        A, B = self.EOS.coefficients_cubic_EOS_vectorized(xij, P)
        Z = self.EOS.Z_vectorized(A, B, ph)
        dAdP = self.EOS.dA_dP()
        dBdP = self.EOS.dB_dP()
        dZdP = self.EOS.dZ_dP_parcial(dAdP, dBdP, Z, A, B)
        dlnphidP = self.EOS.dlnphi_dP(dAdP, dBdP, dZdP, Z, A, B)
        return dlnphidP

    def bubble_point_pressure(self, fprop, ponteiro):
        ponteiro_save = np.copy(ponteiro)
        self.x[:,ponteiro] = fprop.z[:,ponteiro] #self.x[:,ponteiro]
        y = np.copy(self.x)

        # Depende muito de Pbguess (chute inicial de Pb) - PROBLEMÃO AQUI
        #self.Pv[fprop.T > ctes.Tc] = self.Pv[fprop.T > ctes.Tc] * 0.62
        i = 0

        if ctes.Nc>1:
            Pb = 9.5e6*np.ones(len(fprop.P))#np.sum(fprop.z * self.Pv[:,np.newaxis], axis = 0) * 0.62

            K = np.exp(5.37 * (1 + ctes.w) * (1 - 1 / (fprop.T / ctes.Tc)), dtype=np.double)[:,np.newaxis] / \
                    (Pb / ctes.Pc[:,np.newaxis])
            while any(ponteiro):

                y[:,ponteiro] = self.x[:,ponteiro] * K[:,ponteiro]
                Pb_old = np.copy(Pb[ponteiro])

                lnphiv = self.lnphi_based_on_deltaG(y[:,ponteiro], Pb[ponteiro], self.ph_V[ponteiro])
                lnphil = self.lnphi_based_on_deltaG(self.x[:,ponteiro], Pb[ponteiro], self.ph_L[ponteiro])

                fil = np.exp(lnphil) * (self.x[:,ponteiro] * Pb[ponteiro][np.newaxis,:])

                phiv = np.exp(lnphiv)
                phil = np.exp(lnphil)

                dlnphildP = self.get_dlnphidP(fprop.T, self.x[:,ponteiro], Pb[ponteiro], self.ph_L[ponteiro])
                dlnphivdP = self.get_dlnphidP(fprop.T, y[:,ponteiro], Pb[ponteiro], self.ph_V[ponteiro])

                dlnfildP = self.EOS.dlnfij_dP(Pb[ponteiro], dlnphildP)
                dfildP = fil * dlnfildP
                dphivdP = phiv * dlnphivdP

                f = np.sum(fil/phiv, axis = 0) - Pb[ponteiro]
                df = np.sum((phiv * dfildP - fil * dphivdP) / phiv**2, axis=0) - 1.

                i += 1
                if i > 100  or any(df == 0):
                    if any(fprop.P[ponteiro] == fprop.P[19]): ponteiro[19] = False; Pb[19] = 1.1*fprop.P[19]
                    #raise ValueError('Change Pguess - not converging')

                Pb[ponteiro] = Pb[ponteiro] - f / df
                K[:,ponteiro] = phil / phiv
                stop_criteria = abs(Pb[ponteiro] - Pb_old)
                ponteiro_aux = ponteiro[ponteiro]
                ponteiro_aux[stop_criteria <= .5*6894.757] = False
                ponteiro[ponteiro] = ponteiro_aux
        else: Pb = self.Pv * np.ones(len(fprop.P))
        L = fprop.L[ponteiro_save]
        V = fprop.V[ponteiro_save]
        L[fprop.P[ponteiro_save] > Pb[ponteiro_save]] = 1
        V[fprop.P[ponteiro_save] > Pb[ponteiro_save]] = 0.
        L[fprop.P[ponteiro_save] < Pb[ponteiro_save]] = 0.
        V[fprop.P[ponteiro_save] < Pb[ponteiro_save]] = 1
        fprop.L[ponteiro_save] = L
        fprop.V[ponteiro_save] = V

    def update_EOS_dependent_properties(self, fprop):
        #self.EOS = ctes.EOS_class(self.P, fprop.T)

        fprop.phase_molar_densities[0,0,:], fprop.phase_densities[0,0,:] = self.get_EOS_dependent_properties(fprop.T, self.x, fprop.P, self.ph_L)
        fprop.phase_molar_densities[0,1,:], fprop.phase_densities[0,1,:] = self.get_EOS_dependent_properties(fprop.T, self.y, fprop.P, self.ph_V)

    def get_EOS_dependent_properties(self, T, l, P, ph):
        #l - any phase molar composition

        A, B = ctes.EOS_class(T).coefficients_cubic_EOS_vectorized(l, P)
        ph = self.deltaG_molar_vectorized(l, P, ph)
        Z = self.EOS.Z_vectorized(A, B, ph)
        v = Z * ctes.R * T / P #vshift go here
        ksi_phase = 1 / v
        Mw_phase = np.sum(l * ctes.Mw[:,np.newaxis], axis = 0)
        rho_phase = ksi_phase * Mw_phase
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
