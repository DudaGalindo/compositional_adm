from packs.utils.info_manager import InfoManager
from packs.compositional.properties_calculation import PropertiesCalc
from packs.compositional.stability_check import StabilityCheck
from packs.directories import data_loaded
from packs.utils import constants as ctes
import numpy as np
import os

dd = InfoManager('input_cards/inputs_compositional.yml', 'input_cards/inputs0.yml')
dd2 = InfoManager('input_cards/variable_inputs_compositional.yml','input_cards/variable_input.yml')
dd['load_data'] = True
dd.save_obj()
dd2.save_obj()

if dd['deletar_results']:

    results = 'results'
    ff = os.listdir(results)

    for f in ff:
        if f[-4:] == '.vtk':
            os.remove(os.path.join(results, f))

class FluidProperties:
    def __init__(self):
        P = np.array(data_loaded['Pressure']['r1']['value']).astype(float)
        self.P = P*np.ones(ctes.n_volumes)
        self.T = data_loaded['Temperature']['r1']['value']
        self.component_molar_fractions = np.zeros([ctes.n_components, ctes.n_phases, ctes.n_volumes])
        self.phase_molar_densities = np.empty([1, ctes.n_phases, ctes.n_volumes])
        self.phase_densities = np.empty(self.phase_molar_densities.shape)
        if ctes.load_k:
            self.z = np.array([data_loaded['compositional_data']['component_data']['z']]).astype(float) \
                    * np.ones([ctes.Nc, ctes.n_volumes])
        else: z = []

    def inputs_fluid_properties(self, x, y, L, V, ksi_L, ksi_V, rho_L, rho_V):
        self.z = self.z * np.ones(ctes.n_volumes)
        self.component_molar_fractions[0:ctes.Nc,0,:] = x[:,np.newaxis] * np.ones([ctes.Nc, ctes.n_volumes])
        self.component_molar_fractions[0:ctes.Nc,1,:] = y[:,np.newaxis] * np.ones([ctes.Nc, ctes.n_volumes])
        self.L = L * np.ones(ctes.n_volumes)
        self.V = V * np.ones(ctes.n_volumes)
        self.phase_molar_densities[0,0,:] = ksi_L * np.ones(ctes.n_volumes)
        self.phase_molar_densities[0,1,:] = ksi_V * np.ones(ctes.n_volumes)
        self.phase_densities[0,0,:] = rho_L * np.ones(ctes.n_volumes)
        self.phase_densities[0,1,:] = rho_V * np.ones(ctes.n_volumes)

    def inputs_water_properties(self):
        self.phase_densities[0,ctes.n_phases-1,:] = data_loaded['compositional_data']['water_data']['rho_W']
        self.ksi_W0 = self.phase_densities[0,ctes.n_phases-1,:] / ctes.Mw_w
        self.phase_molar_densities[0,ctes.n_phases-1,:] = self.ksi_W0
        self.component_molar_fractions[ctes.n_components-1,ctes.n_phases-1,:] = 1
        #coef_vc7 = np.array([21.573, 0.015122, -27.6563, 0.070615])
        #ctes.vc[ctes.C7 == 1] =  coef_vc7[0] + coef_vc7[1] * np.mean(ctes.Mw[ctes.C7 == 1]) + \
                            #coef_vc7[2] * np.mean([ctes.SG[ctes.C7 == 1]]) + coef_vc7[3] \
                            #* np.mean(ctes.Mw[ctes.C7 == 1]) * np.mean([ctes.SG[ctes.C7 == 1]])
    def update_fluid_properties(self, x, y, L, V, ksi_L, ksi_V, rho_L, rho_V):
        self.component_molar_fractions[0:ctes.Nc,0,:] = x[:,np.newaxis]
        self.component_molar_fractions[0:ctes.Nc,1,:] = y[:,np.newaxis]
        self.L = L
        self.V = V
        self.phase_molar_densities[0,0,:] = ksi_L
        self.phase_molar_densities[0,1,:] = ksi_V
        self.phase_densities[0,0,:] = rho_L
        self.phase_densities[0,1,:] = rho_V
