import numpy as np
from numba import jitclass, float64, int64, experimental
import numba


spec = [('P',float64), ('T', float64), ('w', float64[:]), ('Tc', float64[:]), ('Pc', float64[:]), ('Bin', float64[:,:]), ('Nc', int64), ('K', float64[:]), ('b', float64[:]), ('aalpha_ij', float64[:,:]), ('R', float64), ('aalpha', float64), ('bm', float64), ('psi',float64[:])]
@experimental.jitclass(spec)
class PengRobinson(object):
    def __init__(self, P, T,  w, Tc, Pc, Bin):
        self.P = P
        self.T = T
        self.Tc = Tc
        self.Pc = Pc
        self.Bin = Bin
        self.w = w
        self.Nc = len(w)
        self.b = np.zeros(self.Nc)
        self.aalpha_ij = np.zeros(self.Bin.shape)
        self.aalpha = 0.
        self.bm = 0.
        self.psi = np.zeros(self.Nc)
        self.R = 8.3144598
        self.coefficientsPR()

    def coefficientsPR(self):
        #l - any phase molar composition
        PR_kC7 = np.array([0.379642, 1.48503, 0.1644, 0.016667])
        PR_k = np.array([0.37464, 1.54226, 0.26992])

        k = (PR_kC7[0] + PR_kC7[1] * self.w - PR_kC7[2] * self.w ** 2 + \
            PR_kC7[3] * self.w ** 3) * (1*(self.w >= 0.49))  + (PR_k[0] + PR_k[1] * self.w - \
            PR_k[2] * self.w ** 2) * (1*(self.w < 0.49))
        alpha = (1 + k * (1 - (self.T/ self.Tc) ** (1 / 2))) ** 2
        aalpha_i = 0.45724 * (self.R * self.Tc) ** 2 / self.Pc * alpha
        self.b = 0.07780 * self.R * self.Tc / self.Pc
        #aalpha_i_reshape = np.ones(self.Bin.shape) * aalpha_i[:,np.newaxis]
        for i in range(self.Nc):
            for j in range(self.Nc):
                self.aalpha_ij[i,j] = np.sqrt(aalpha_i[i] * aalpha_i[j]) * (1 - self.Bin[i,j])


    def coefficients_cubic_EOS(self, l):
        #self.b, self.aalphaij = self.coefficientsPR()
        self.aalpha = 0.
        self.psi = np.zeros(self.Nc)
        self.bm = np.sum(l * self.b)
        #l_reshape = np.ones((self.aalpha_ij).shape) * l[:, np.newaxis]
        for i in range(self.Nc):
            for j in range(self.Nc):
                self.aalpha += (l[i] * l[j] * self.aalpha_ij[i,j])
                self.psi[i] += l[j] * self.aalpha_ij[i,j]
        B = self.bm * self.P/ (self.R* self.T)
        A = self.aalpha * self.P/ (self.R* self.T) ** 2
        #self.psi = (l_reshape * self.aalpha_ij).sum(axis = 0)
        return A, B

    def Z(self, A, B):
        # PR cubic EOS: Z**3 - (1-B)*Z**2 + (A-2*B-3*B**2)*Z-(A*B-B**2-B**3)
        coef = np.array([1, -(1 - B), (A - 2*B - 3*B**2), -(A*B - B**2 - B**3)], dtype=np.complex_)
        Z = np.roots(coef)
        reais = np.argwhere(np.abs(np.imag(Z)) <1e-15 )
        Z_reais = np.real(Z)[reais.ravel()]
        #root = np.isreal(Z) # return True for real roots
        #position where the real roots are - crated for organization only
        #real_roots_position = np.where(root == True)
        #Z_reais = np.real(Z[real_roots_position[:]]) #Saving the real roots
        #Z_ans = min(Z_reais) * ph + max(Z_reais) * (1 - ph)
        return Z_reais
        ''' This last line, considers that the phase is composed by a pure
         component, so the EOS model can return more than one real root.
            If liquid, Zl = min(Z) and gas, Zv = max(Z).
            You can notice that, if there's only one real root,
            it works as well.'''


    def lnphi(self, l, ph):
        #l - any phase molar composition
        #l = l[:,np.newaxis]
        A, B = self.coefficients_cubic_EOS(l)
        Z = self.Z(A, B)
        Z = np.min(Z) * ph + np.max(Z) * (1 - ph)
        lnphi = self.b / self.bm * (Z - 1) - np.log(Z - B) - A / (2 * (2 ** (1/2))* B) * (2 * self.psi / self.aalpha - self.b / self.bm) * np.log((Z + (1 + 2 ** (1/2)) * B) / (Z + (1 - 2 ** (1/2)) * B))

        return lnphi

    """ Bellow is showed two functions of the PR EOS equation, that were constructed in a vectorized manner.
     I modified the code so this two functions are used in the vectorized and non vectorized part. For now they
     are working in both of them. But, just because I didn't tested everything possible, I don't completelly
     trust them. Thats why they are kind of separeted from the original ones(that I do trust completelly but
     you can't calculate in a vectorized manner with them."""

    def coefficients_cubic_EOS_vectorized(self, l):
        self.bm = np.sum(l * self.b[:,np.newaxis], axis=0)
        l_reshape = np.ones((self.aalpha_ij).shape)[:,:,np.newaxis] * l[:,np.newaxis,:]
        self.aalpha = (l_reshape * l[np.newaxis,:,:] * self.aalpha_ij[:,:,np.newaxis]).sum(axis=0).sum(axis=0)
        B = self.bm * self.P / (ctes.R* self.T)
        A = self.aalpha * self.P / (ctes.R* self.T) ** 2
        self.psi = (l_reshape * self.aalpha_ij).sum(axis = 0)
        return A, B

    def Z_vectorized(self, A, B):
        coef = np.empty([4,len(B.ravel())])
        coef[0,:] = np.ones(len(B))
        coef[1,:] = -(1 - B)
        coef[2,:] = (A - 2*B - 3*B**2)
        coef[3,:] = -(A*B - B**2 - B**3)
        Z = CubicRoots().run(coef)
        return Z

    """ Derivatives - Still need to organize this"""

    def get_dVt_dNk_analytically(self, P, Vt, Sj, l, Nk):
        self.coefficients_cubic_EOS_vectorized(l)
        bm = self.bm; am = self.aalpha
        C = np.array([P*(Sj*l)**3, (bm*P - ctes.R*self.T)*(Sj*l)**2, (am - 3*P*bm**2 - 2*ctes.R*self.T*bm)*(Sj*l)])

        Num = 3*Vt**3/(Nk**4)*C[0] + 2*Vt**2/(Nk**3)*C[1] + Vt/(Nk**2)*C[2]
        Den = 3*Vt**2/(Nk**3)*C[0] + 2*Vt/(Nk**2)*C[1] + 1/Nk*C[2]

        dVt_dNk = Num/Den
        return dVt_dNk

    def get_dVt_dP_analytically(self, P, Vt, Nj, l):
        self.coefficients_cubic_EOS_vectorized(l)
        bm = self.bm; am = self.aalpha
        C = np.array([(1/Nj)**3, (1/Nj)**2, (1/Nj)])

        Num = - Vt**3*C[0] - bm**3 - Vt**2*bm*C[1] + 3*bm**2*Vt*C[2]
        Den = 3*Vt**2*P*C[0] + 2*Vt*(bm*P - ctes.R*self.T)*C[1] + (am - 3*bm**2*P - 2*ctes.R*self.T*bm)*C[2]

        dVt_dP = Num/Den
        return dVt_dP

    def get_all_derivatives(self, fprop):
        n_blocks = len(fprop.P)

        P = fprop.P
        So = fprop.So
        Vt = fprop.Vt
        l = fprop.component_molar_fractions[0:self.Nc,0,:]
        Nk = fprop.component_mole_numbers[0:self.Nc,:]
        dVt_dNk = self.get_dVt_dNk_analytically(P, Vt, So, l, Nk)
        No = fprop.phase_mole_numbers[0,0,:]
        dVt_dP = self.get_dVt_dP_analytically(P, Vt, No, l)
        return dVt_dNk, dVt_dP
