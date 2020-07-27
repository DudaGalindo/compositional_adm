import numpy as np
from ..directories import data_loaded
from ..utils import constants as ctes
from .stability_check import StabilityCheck
from .properties_calculation import PropertiesCalc
import scipy.sparse as sp
from scipy import linalg

class FOUM:

    def update_flux(self, fprop, total_flux_internal_faces):
        self.update_phase_flux_internal_faces(fprop, total_flux_internal_faces)
        self.update_flux_volumes(fprop)

    def update_phase_flux_internal_faces(self, fprop, total_flux_internal_faces):
        z = ctes.z[ctes.v0[:,0]]
        z_up = ctes.z[ctes.v0[:,1]]

        frj = fprop.mobilities_internal_faces[0,:,:] / np.sum(fprop.mobilities_internal_faces[0,:,:], axis = 0)

        self.phase_flux_internal_faces = frj[np.newaxis,:] * (total_flux_internal_faces +
                                        ctes.pretransmissibility_internal_faces * (np.sum(fprop.mobilities_internal_faces *
                                        (fprop.Pcap[:,ctes.v0[:,1]] - fprop.Pcap[:,ctes.v0[:,0]] - ctes.g *
                                        fprop.phase_densities_internal_faces * (z_up - z)), axis=1) - \
                                        np.sum(fprop.mobilities_internal_faces, axis=1) * (fprop.Pcap[:,ctes.v0[:,1]] -
                                        fprop.Pcap[:,ctes.v0[:,0]] - ctes.g *fprop.phase_densities_internal_faces * (z_up - z))))

        # M.flux_faces[M.faces.internal] = total_flux_internal_faces * M.faces.normal[M.faces.internal].T

    def update_flux_volumes(self, fprop):
        component_flux_internal_faces = np.sum(fprop.component_molar_fractions_internal_faces * fprop.phase_molar_densities_internal_faces *
                                self.phase_flux_internal_faces, axis = 1)
        cx = np.arange(ctes.n_components)
        lines = np.array([np.repeat(cx,len(ctes.v0[:,0])), np.repeat(cx,len(ctes.v0[:,1]))]).astype(int).flatten()
        cols = np.array([np.tile(ctes.v0[:,0],ctes.n_components), np.tile(ctes.v0[:,1], ctes.n_components)]).flatten()
        data = np.array([-component_flux_internal_faces, component_flux_internal_faces]).flatten()
        fprop.component_flux_vols_total = sp.csc_matrix((data, (lines, cols)), shape = (ctes.n_components, ctes.n_volumes)).toarray()

        
class MUSCL:

    """Class created for the second order MUSCL implementation for the calculation of the advective terms"""

    def __init__(self, M, fprop, wells):
        dNk_vols = self.volume_gradient_reconstruction(M, fprop, wells)
        dNk_face, dNk_face_neig = self.get_faces_gradient(M, fprop, dNk_vols)
        phi = self.Van_Leer_slope_limiter(dNk_face, dNk_face_neig)
        Nk_face, z_face = self.get_extrapolated_compositions(fprop, phi, dNk_face_neig)
        self.get_extrapolated_properties(fprop, M, Nk_face, z_face)
        G = self.update_gravity_term()
        update_flux_Roe(self, fprop, G)

    def volume_gradient_reconstruction(self, M, fprop, wells):
        neig_vols = M.volumes.bridge_adjacencies(M.volumes.all,2,3)
        matriz = np.zeros((ctes.n_volumes,ctes.n_volumes))

        lines = np.array([ctes.v0[:, 0], ctes.v0[:, 1], ctes.v0[:, 0], ctes.v0[:, 1]]).flatten()
        cols = np.array([ctes.v0[:, 1], ctes.v0[:, 0], ctes.v0[:, 0], ctes.v0[:, 1]]).flatten()
        data = np.array([np.ones(len(ctes.v0[:, 0])), np.ones(len(ctes.v0[:, 0])),
                        np.zeros(len(ctes.v0[:, 0])), np.zeros(len(ctes.v0[:, 0]))]).flatten()
        all_neig = sp.csc_matrix((data, (lines, cols)), shape = (ctes.n_volumes, ctes.n_volumes)).toarray()
        all_neig2 = all_neig + np.identity(ctes.n_volumes)

        Nk_neig =  fprop.component_mole_numbers[:,np.newaxis,:] * all_neig2[np.newaxis,:,:]
        Nk = Nk_neig.transpose(0,2,1)
        pos_neig = M.data['centroid_volumes'].T[:,np.newaxis,:] * all_neig2[np.newaxis,:,:]
        pos = pos_neig.transpose(0,2,1)

        ds = pos_neig - pos
        ds_norm = np.linalg.norm(ds, axis=0)
        versor_ds = np.empty(ds.shape)
        versor_ds[:,ds_norm==0] = 0
        versor_ds[:,ds_norm!=0] = ds[:,ds_norm!=0] / ds_norm[ds_norm!=0]
        dNk = Nk_neig - Nk

        dNk_by_axes = np.repeat(dNk[:,np.newaxis,:,:],3,axis=1)
        dNk_by_axes = dNk_by_axes * versor_ds
        dNk_vols = dNk_by_axes.sum(axis = 3)

        ds_vols = ds * versor_ds
        ds_vols = ds_vols.sum(axis = 2)
        dNkds_vols = np.copy(dNk_vols)
        dNkds_vols[:,ds_vols!=0] = dNk_vols[:,ds_vols != 0] / ds_vols[ds_vols != 0][np.newaxis,:]
        return dNkds_vols

    def get_faces_gradient(self, M, fprop, dNkds_vols):
        dNk_face =  fprop.component_mole_numbers[:,ctes.v0[:,1]] - fprop.component_mole_numbers[:,ctes.v0[:,0]]
        ds_face = M.data['centroid_volumes'][ctes.v0[:,1],:] -  M.data['centroid_volumes'][ctes.v0[:,0],:]
        dNk_face_vols = 2 * (dNkds_vols[:,:,ctes.v0] * ds_face.T[np.newaxis,:,:,np.newaxis]).sum(axis=1)
        dNk_face_neig = dNk_face_vols - dNk_face[:,:,np.newaxis]
        return dNk_face, dNk_face_neig

    def Van_Leer_slope_limiter(self, dNk_face, dNk_face_neig):
        r_face = dNk_face[:,:,np.newaxis] / dNk_face_neig
        r_face[dNk_face_neig==0] = 0
        phi = (r_face + abs(r_face)) / (r_face + 1)
        phi[:,:,1] = -phi[:,:,1]
        return phi

    def get_extrapolated_compositions(self, fprop, phi, dNk_face_neig):
        Nk_face = fprop.component_mole_numbers[:,ctes.v0] + phi / 2 * dNk_face_neig
        z_face = Nk_face[0:ctes.n_components-1] / np.sum(Nk_face[0:ctes.n_components-1], axis = 1)[:,np.newaxis,:]
        return Nk_face, z_face

    def get_extrapolated_properties(self, fprop, M, Nk_face, z_face):
        import pdb; pdb.set_trace()
        L_face = np.empty((len(M.faces.internal), 2))
        V_face = np.empty((len(M.faces.internal), 2))
        Sw_face = np.empty((len(M.faces.internal), 2))
        So_face = np.empty((len(M.faces.internal), 2))
        Sg_face = np.empty((len(M.faces.internal), 2))
        self.component_molar_fractions_face = np.empty((ctes.Nc, ctes.n_phases, len(M.faces.internal), 2))
        self.phase_molar_densities_face = np.empty((0, ctes.n_phases, len(M.faces.internal), 2))
        self.phase_densities_face = np.empty((0, ctes.n_phases, len(M.faces.internal), 2))
        self.mobilities_face = np.empty((0, ctes.n_phases, len(M.faces.internal), 2))

        self.P_face = np.sum(fprop.P[ctes.v0[:,0]], axis=1) * 0.5
        for i in range(2):
            L_face[:,i], V_face[:,i], self.component_molar_fractions_face[0:ctes.Nc,0,:,i], \
            self.component_molar_fractions_face[0:ctes.Nc,1,:,i], self.phase_molar_densities_face[:,0,:,i], \
            self.phase_molar_densities_face[:,1,:,i], self.phase_densities_face[:,0,:,i], \
            self.phase_densities_face[:,1,:,i]  =  StabilityCheck(fprop, self.P_face).run(fprop, self.P_face, \
                                                            L_face[:,i], V_face[:,i], z_face[:,i])

            Sw_face[:,i] = PropertiesCalc().update_water_saturation(fprop, Nk_face[:,i])
            So_face[:,i], Sg_face[:,i] =  PropertiesCalc().update_saturations(Sw_face[:,i],
                                self.phase_molar_densities_face[:,:,:,i], L_face[:,i], V_face[:,i])

            self.mobilities_face[:,:,:,i] = PropertiesCalc().update_mobilities(fprop, So_face[:,i], Sg_face[:,i],
                                        Sw_face[:,i], self.phase_molar_densities_face[:,:,:,i],
                                        self.component_molar_fractions_face[:,:,:,i])

    def update_gravity_term(self):
        G = ctes.g * self.phase_densities_face * ctes.z
        return G

    def get_faces_properties_upwind(self, fprop, G):

        ''' Using one-point upwind approximation '''

        Pot_hidj = self.P_face + fprop.Pcap[:,ctes.v0[:,0]] - G[0,:,ctes.v0[:,0]]
        Pot_hidj_up = self.P_face + fprop.Pcap[:,ctes.v0[:,0]] - G[0,:,ctes.v0[:,0]]

        fprop.mobilities_internal_faces = np.zeros([1, ctes.n_phases, ctes.n_internal_faces])
        mobilities_vols = self.mobilities_faces[:,:,:,0]
        mobilities_vols_up = self.mobilities_faces[:,:,:,1]
        fprop.mobilities_internal_faces[0,Pot_hidj_up <= Pot_hidj] = mobilities_vols[0,Pot_hidj_up <= Pot_hidj]
        fprop.mobilities_internal_faces[0,Pot_hidj_up > Pot_hidj] = mobilities_vols_up[0,Pot_hidj_up > Pot_hidj]

        fprop.phase_molar_densities_internal_faces = np.zeros([1, ctes.n_phases, ctes.n_internal_faces])
        phase_molar_densities_vols = self.phase_molar_densities_face[:,:,:,0]
        phase_molar_densities_vols_up = self.phase_molar_densities_face[:,:,:,1]
        fprop.phase_molar_densities_internal_faces[0,Pot_hidj_up <= Pot_hidj] = phase_molar_densities_vols[0,Pot_hidj_up <= Pot_hidj]
        fprop.phase_molar_densities_internal_faces[0,Pot_hidj_up > Pot_hidj] = phase_molar_densities_vols_up[0,Pot_hidj_up > Pot_hidj]

        fprop.component_molar_fractions_internal_faces = np.zeros([ctes.n_components, ctes.n_phases, ctes.n_internal_faces])
        component_molar_fractions_vols = self.component_molar_fractions_face[:,:,:,0]
        component_molar_fractions_vols_up = self.component_molar_fractions_face[:,:,:,1]
        fprop.component_molar_fractions_internal_faces[:,Pot_hidj_up <= Pot_hidj] = component_molar_fractions_vols[:,Pot_hidj_up <= Pot_hidj]
        fprop.component_molar_fractions_internal_faces[:,Pot_hidj_up > Pot_hidj] = component_molar_fractions_vols_up[:,Pot_hidj_up > Pot_hidj]

    def get_phase_densities_internal_faces(self, fprop):
        fprop.phase_densities_internal_faces = (fprop.Vp[ctes.v0[:,0]] * self.phase_densities_face[:,:,:,0] +
                                                fprop.Vp[ctes.v0[:,1]] * self.phase_densities_face[:,:,:,1]) /  \
                                                (fprop.Vp[ctes.v0[:,0]] + fprop.Vp[ctes.v0[:,1]])

    def update_total_flux_internal_faces(self, M, fprop):
        Pot_hidj = self.P_face[:,0] + fprop.Pcap[:,ctes.v0[:,0]]
        Pot_hidj_up = self.P_face[:,1] + fprop.Pcap[:,ctes.v0[:,1]]
        z = ctes.z[ctes.v0[:,0]]
        z_up = ctes.z[ctes.v0[:,1]]
        self.total_flux_internal_faces = - np.sum(fprop.mobilities_internal_faces * ctes.pretransmissibility_internal_faces
                                         * ((Pot_hidj_up - Pot_hidj) - ctes.g * fprop.phase_densities_internal_faces
                                         * (z_up - z)), axis = 1)
        self.total_flux_internal_faces_LLF = - np.sum(self.mobilities_faces * ctes.pretransmissibility_internal_faces
                                         * ((Pot_hidj_up - Pot_hidj) - ctes.g * self.phase_densities_faces
                                         * (z_up - z)), axis = 1)

    def update_flux_Roe(self, fprop, G):
        self.get_faces_properties_upwind(fprop, G)
        self.get_phase_densities_internal_faces(fprop)
        FOUM(fprop, self.total_flux_internal_faces)

    def update_flux_LLF(self):

        phase_flux_faces_LLF = np.sum(self.total_flux_internal_faces_LLF, axis=3)
        self.phase_flux_internal_faces = np.sum(phase_flux_faces_LLF,axis=3)
