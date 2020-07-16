import numpy as np
from ..directories import data_loaded
from ..utils import constants as ctes
import scipy.sparse as sp
from scipy import linalg

class FOUM:

    def update_flux(self, fprop, total_flux_internal_faces):
        self.update_phase_flux_internal_faces(fprop, total_flux_internal_faces)
        self.update_flux_volumes(fprop)

    def update_phase_flux_internal_faces(self, fprop, total_flux_internal_faces):
        z = ctes.z[ctes.v0[:,0]]
        z_up = ctes.z[ctes.v0[:,1]]
        #Pot_hid = fprop.P + fprop.Pcap
        #Pot_hidj = Pot_hid[:,ctes.v0[:,0]]
        #Pot_hidj_up = Pot_hid[:,ctes.v0[:,1]]

        frj = fprop.mobilities_internal_faces[0,:,:] / np.sum(fprop.mobilities_internal_faces[0,:,:], axis = 0)

        self.phase_flux_internal_faces = frj * (total_flux_internal_faces + (np.sum(fprop.mobilities_internal_faces *
                                        ctes.pretransmissibility_internal_faces * (fprop.Pcap[:,ctes.v0[:,1]] -
                                        fprop.Pcap[:,ctes.v0[:,0]] - ctes.g * fprop.phase_densities_internal_faces *
                                        (z_up - z)), axis=1) - np.sum(fprop.mobilities_internal_faces *
                                         ctes.pretransmissibility_internal_faces, axis=1) * (fprop.Pcap[:,ctes.v0[:,1]] -
                                         fprop.Pcap[:,ctes.v0[:,0]] - ctes.g *fprop.phase_densities_internal_faces * (z_up - z))))

        #phase_flux_internal_faces = - (fprop.mobilities_internal_faces * ctes.pretransmissibility_internal_faces
        #                             * (Pot_hidj_up - Pot_hidj - ctes.g * fprop.phase_densities_internal_faces
        #                             * (z_up - z)))
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
        Nk_face = self.get_extrapolated_compositions(fprop, phi, dNk_face_neig)


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
        dNk_vols = dNk_by_axes.sum(axis=3)

        ds_vols = ds * versor_ds
        ds_vols = ds_vols.sum(axis=2)
        dNkds_vols = np.copy(dNk_vols)
        dNkds_vols[:,ds_vols!=0] = dNk_vols[:,ds_vols!=0] / ds_vols[ds_vols!=0][np.newaxis,:]
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
        return Nk_face

    def get_extrapolated_properties():
        #mobilities, do flash again (see how I am going to do that), and get with flash new ksi, x and y.
        pass

    def update_flux_upwind():
        pass

    def update_flux_LLF():
        pass
