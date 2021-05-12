import numpy as np
from ..properties_calculation import PropertiesCalc
from ..IMPEC.flux_calculation import Flux

class saturation:
    def __init__(self):
        pass

    def update_pore_volume(self):
        Vp = PropertiesCalc().update_porous_volume(P)
        return Vp

    def update_mobilities(self, fprop, Sj):
        So = Sj[0]
        Sg = Sj[1]
        Sw = 1 - So - Sg
        krs_new = PropertiesCalc().update_relative_permeabilities(fprop, So, Sg, Sw)
        mobilities = krs_new/fprop.phase_viscosity
        return mobilities

    def mobilities_faces_upwind(self, P_new, mobilities):
        Pot_hid = P_new + fprop.Pcap - self.G[0,:,:]
        Pot_hidj = Pot_hid[:,ctes.v0[:,0]]
        Pot_hidj_up = Pot_hid[:,ctes.v0[:,1]]

        mobilities_internal_faces = np.zeros([1, ctes.n_phases, ctes.n_internal_faces])
        mobilities_vols = mobilities[:,:,ctes.v0[:,0]]
        mobilities_vols_up = mobilities[:,:,ctes.v0[:,1]]
        mobilities_internal_faces[0,Pot_hidj_up <= Pot_hidj] = mobilities_vols[0,Pot_hidj_up <= Pot_hidj]
        mobilities_internal_faces[0,Pot_hidj_up > Pot_hidj] = mobilities_vols_up[0,Pot_hidj_up > Pot_hidj]
        return mobilities_internal_faces

    def update_phase_flux(self, M, fprop, P_old, Ft_internal_faces, mobilities_internal_faces_new):
        Fk_vols_total_new, wave_velocity = Flux().update_flux(M, fprop, P_old, Ft_internal_faces,
                             fprop.rho_j_internal_faces, mobilities_internal_faces_new)
        Fk_vols_total_new = Fk_vols_total_new[:-1] #except water component/phase
        return Fk_vols_total_new

    def update_well_term(self, fprop, qk, mobilities):
        wp = wells['ws_p']
        if len(wp)>0:
            mob_ratio = mobilities[:,:,wp] / np.sum(mobilities[:,:,wp], axis = 1)
            qj = mob_ratio * np.sum(fprop.q_phase, axis=1)
            qk[:,wp] = np.sum(fprop.xkj[:,:,wp] * fprop.Csi_j[:,:,wp] * qj, axis = 1)
        return qk

    def implicit_solver(self, M, fprop, Ft_internal_faces, Vp_new, dVjdP, P_new, P_old, qk, delta_t):
        Vj = fprop.Nj[0,0:2] / fprop.Csi_j[0,0:2] #only hydrocarbon phases
        dVjdP = dVjdP[0,0:2,:] #only hydrocarbon phases
        dVjdNk = dVjdNk[:-1,0:2,:] #only hydrocarbon phases
        q_total = fprop.q_phase.sum(axis=1) #for the producer well
        S_new = p

        while any(ponteiro.ravel()):
            mobilities_new = self.update_mobilities(fprop, S_new)
            mobilities_internal_faces = self.mobilities_faces_upwind(P_new, mobilities_new)
            Fk_vols_total_new = self.update_phase_flux(M, fprop, P_old, Ft_internal_faces, mobilities_internal_faces_new)
            qk_new = self.update_well_term(fprop, np.copy(qk), mobilities)
            Rj = S_new * Vp_new - Vj - dVjdP * (P_new - P_old) - delta_t * np.sum(dVjdNk * (Fk_vols_total_new +
                qk_new), axis=0)
