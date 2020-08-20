import numpy as np
from ..directories import data_loaded
from ..utils import constants as ctes
from ..solvers.solvers_scipy.solver_sp import SolverSp
from scipy import linalg
import scipy.sparse as sp
from . import equation_of_state

class TPFASolver:
    def __init__(self, fprop):
        self.dVt_derivatives(fprop)

    def get_pressure(self, M, wells, fprop, delta_t):
        T = self.update_transmissibility(M, wells, fprop, delta_t)
        D = self.update_independent_terms(M, fprop, wells, delta_t)
        self.update_pressure(T, D, fprop)
        self.update_total_flux_internal_faces(M, fprop)
        self.update_flux_wells(fprop, wells, D, delta_t)
        return self.P, self.total_flux_internal_faces, self.q

    def dVt_derivatives(self, fprop):
        self.dVtk = np.empty([ctes.n_components, ctes.n_volumes])

        if ctes.load_k:
            self.EOS = ctes.EOS_class(fprop.T)
            if not ctes.compressible_k:
                dVtP = np.zeros(ctes.n_volumes)
                self.dVtk[0:ctes.Nc,:] = 1 / fprop.phase_molar_densities[0,0,:]
            else: self.dVtk[0:ctes.Nc,:], dVtP = self.EOS.get_all_derivatives(fprop)

        else: dVtP = np.zeros(ctes.n_volumes)

        if ctes.load_w:

            self.dVtk[ctes.n_components-1,:] = 1 / fprop.phase_molar_densities[0,ctes.n_phases-1,:]
            dVwP = - fprop.component_mole_numbers[ctes.Nc,:] * fprop.ksi_W0 * ctes.Cw / (fprop.ksi_W)**2

        else: dVwP = np.zeros(ctes.n_volumes)

        self.dVtP = dVtP + dVwP
        #self.dVtP = - 1.5083925365317445e-09 * fprop.Vp

    def update_transmissibility(self, M, wells, fprop, delta_t):
        self.t0_internal_faces_prod = fprop.component_molar_fractions_internal_faces * \
                                      fprop.phase_molar_densities_internal_faces * \
                                      fprop.mobilities_internal_faces

        ''' Transmissibility '''
        t0 = (self.t0_internal_faces_prod).sum(axis = 1)
        t0 = t0 * ctes.pretransmissibility_internal_faces
        T = np.zeros([ctes.n_volumes, ctes.n_volumes])
        #self.T_noCC_wp = np.zeros()
        # Look for a way of doing this not using a loop
        for i in range(ctes.n_components):
            lines = np.array([ctes.v0[:, 0], ctes.v0[:, 1], ctes.v0[:, 0], ctes.v0[:, 1]]).flatten()
            cols = np.array([ctes.v0[:, 1], ctes.v0[:, 0], ctes.v0[:, 0], ctes.v0[:, 1]]).flatten()
            data = np.array([-t0[i,:], -t0[i,:], +t0[i,:], +t0[i,:]]).flatten()

            Ta = (sp.csc_matrix((data, (lines, cols)), shape = (ctes.n_volumes, ctes.n_volumes))).toarray()
            T += Ta * self.dVtk[i,:, np.newaxis]

        T = T * delta_t
        ''' Transmissibility diagonal term '''
        diag = np.diag((ctes.Vbulk * ctes.porosity * ctes.Cf - self.dVtP))
        T += diag

        self.T_noCC = np.copy(T)
        ''' Includding contour conditions '''
        T[wells['ws_p'],:] = 0
        T[wells['ws_p'], wells['ws_p']] = 1
        return T

    def pressure_independent_term(self, fprop):
        vector = ctes.Vbulk * ctes.porosity * ctes.Cf - self.dVtP
        pressure_term = vector * fprop.P
        return pressure_term

    def capillary_and_gravity_independent_term(self, fprop):

        t0_j = self.t0_internal_faces_prod * ctes.pretransmissibility_internal_faces
        t0_k = ctes.g * np.sum(fprop.phase_densities_internal_faces * t0_j, axis=1)

        # Look for a better way to do this
        cap = np.zeros([ctes.n_volumes])
        grav = np.zeros([ctes.n_volumes,ctes.n_volumes])
        if any((ctes.z - ctes.z[0]) != 0):
            for i in range(ctes.n_components):
                lines = np.array([ctes.v0[:, 0], ctes.v0[:, 1], ctes.v0[:, 0], ctes.v0[:, 1]]).flatten()
                cols = np.array([ctes.v0[:, 1], ctes.v0[:, 0], ctes.v0[:, 0], ctes.v0[:, 1]]).flatten()
                data = np.array([t0_k[i,:], t0_k[i,:], -t0_k[i,:], -t0_k[i,:]]).flatten()
                t0_rho = sp.csc_matrix((data, (lines, cols)), shape = (ctes.n_volumes, ctes.n_volumes)).toarray()
                grav += t0_rho * self.dVtk[i,:]

                for j in range(ctes.n_phases):
                    lines = np.array([ctes.v0[:, 0], ctes.v0[:, 1], ctes.v0[:, 0], ctes.v0[:, 1]]).flatten()
                    cols = np.array([ctes.v0[:, 1], ctes.v0[:, 0], ctes.v0[:, 0], ctes.v0[:, 1]]).flatten()
                    data = np.array([t0_j[i,j,:], t0_j[i,j,:], -t0_j[i,j,:], -t0_j[i,j,:]]).flatten()
                    t0 = sp.csc_matrix((data, (lines, cols)), shape = (ctes.n_volumes, ctes.n_volumes))*self.dVtk[i,:]
                    cap += t0 @ fprop.Pcap[j,:]

        gravity_term = grav @ ctes.z

        # capillary_term = np.sum(self.dVtk * np.sum (fprop.component_molar_fractions *
        #         fprop.phase_molar_densities * fprop.mobilities * fprop.Pcap, axis = 1), axis = 0)
        return cap, gravity_term

    def volume_discrepancy_independent_term(self, fprop):
        volume_discrepancy_term = fprop.Vp - fprop.Vt
        if np.max(abs(fprop.Vp - fprop.Vt)/fprop.Vp) > 1e-3:
            import pdb; pdb.set_trace()
            if fprop.P[2]>fprop.P[1]: import pdb; pdb.set_trace()
            #raise ValueError('diminuir delta_t')
        return volume_discrepancy_term

    def well_term(self, fprop, wells):
        self.q = np.zeros([ctes.n_components, ctes.n_volumes]) #for now
        well_term = np.zeros(ctes.n_volumes)

        '''Modificar isto para o cálculo ser feito apenas uma vez'''
        if len(wells['ws_q']) > 0:
            '''if len(wells['ws_p'])>1:
                bhp_ind = np.argwhere(M.volumes.center[wells['ws_p']][:,2] == min(M.volumes.center[wells['ws_p']][:,2])).ravel()
            else: bhp_ind = wells['ws_p']
            wells['values_p'] = wells['values_p'] + ctes.g * fprop.phase_densities[0,0,wells['ws_p']] * (ctes.z[wells['ws_p']] - ctes.z[bhp_ind])
            '''
            #well_term[wells['ws_q']] = wells['values_q_vol']
            #self.q[:,wells['ws_q']] = wells['values_q'] / 55550 /self.dVtk[:,wells['ws_q']]
            self.q[:,wells['ws_q']] =  wells['values_q']
            well_term[wells['ws_q']] = np.sum(self.dVtk[:,wells['ws_q']] * self.q[:,wells['ws_q']], axis = 0)
            #if fprop.Sw[0]>=0.29: import pdb; pdb.set_trace()
            '''self.q[:,wells['ws_p']] = wells['WI'] * np.sum(fprop.component_molar_fractions[:,:,wells['ws_p'] *
            fprop.phase_molar_densities[:,:,wells['ws_p']] * fprop.mobilities[:,:,wells['ws_p']] *
            wells['values_p']], axis=2)
            well_term[wells['ws_p']] = np.sum(self.q[:,wells['ws_p']] * self.dVtk[:,wells['ws_p']], axis=0)'''
        return well_term

    def update_independent_terms(self, M, fprop, wells, delta_t):
        self.pressure_term = self.pressure_independent_term(fprop)
        self.capillary_term, self.gravity_term = self.capillary_and_gravity_independent_term(fprop)
        self.volume_term = self.volume_discrepancy_independent_term(fprop)
        well_term = self.well_term(fprop, wells)
        independent_terms = self.pressure_term - self.volume_term  +  delta_t * well_term - delta_t * (self.capillary_term + self.gravity_term)
        if len(wells['ws_p'])>1:
            bhp_ind = np.argwhere(M.volumes.center[wells['ws_p']][:,2] == min(M.volumes.center[wells['ws_p']][:,2])).ravel()
        else: bhp_ind = wells['ws_p']
        independent_terms[wells['ws_p']] = wells['values_p'] + ctes.g * fprop.phase_densities[0,0,wells['ws_p']] * (ctes.z[wells['ws_p']] - ctes.z[bhp_ind])
        #import pdb; pdb.set_trace()
        return independent_terms

    def update_pressure(self, T, D, fprop):
        self.P = linalg.solve(T,D)

    def update_total_flux_internal_faces(self, M, fprop):
        Pot_hid = self.P + fprop.Pcap
        Pot_hidj = Pot_hid[:,ctes.v0[:,0]]
        Pot_hidj_up = Pot_hid[:,ctes.v0[:,1]]
        z = ctes.z[ctes.v0[:,0]]
        z_up = ctes.z[ctes.v0[:,1]]
        self.total_flux_internal_faces = - np.sum(fprop.mobilities_internal_faces * ctes.pretransmissibility_internal_faces
                                         * ((Pot_hidj_up - Pot_hidj) - ctes.g * fprop.phase_densities_internal_faces
                                         * (z_up - z)), axis = 1)

    def update_flux_wells(self, fprop, wells, independent_terms, delta_t):
        wp = wells['ws_p']

        if len(wp)>=1:
            well_term =  (self.T_noCC[wp,:] @ self.P - self.pressure_term[wp] + self.volume_term[wp]) / delta_t \
                            + self.capillary_term[wp] + self.gravity_term[wp]
            mob_ratio = fprop.mobilities[:,:,wp] / np.sum(fprop.mobilities[:,:,wp], axis = 1)
            self.q[:,wp] = np.sum(fprop.component_molar_fractions[:,:,wp] * mob_ratio *
                            fprop.phase_molar_densities[:,:,wp] * well_term, axis = 1)
            fprop.q_phase = mob_ratio * well_term
