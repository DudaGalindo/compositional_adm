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
            self.z = np.array([data_loaded['compositional_data']['component_data']['z']]).astype(float)
        else: z = []

    def inputs_fluid_properties(self, fprop_block):
        self.z = self.z * np.ones(ctes.n_volumes)
        self.component_molar_fractions[0:ctes.Nc,0,:] = fprop_block.x[:,np.newaxis] * np.ones([ctes.Nc, ctes.n_volumes]).astype(float)
        self.component_molar_fractions[0:ctes.Nc,1,:] = fprop_block.y[:,np.newaxis] * np.ones([ctes.Nc, ctes.n_volumes]).astype(float)
        self.L = fprop_block.L * np.ones(ctes.n_volumes).astype(float)
        self.V = fprop_block.V * np.ones(ctes.n_volumes).astype(float)
        self.phase_molar_densities[0,0,:] = fprop_block.ksi_L * np.ones(ctes.n_volumes).astype(float)
        self.phase_molar_densities[0,1,:] = fprop_block.ksi_V * np.ones(ctes.n_volumes).astype(float)
        self.phase_densities[0,0,:] = fprop_block.rho_L * np.ones(ctes.n_volumes).astype(float)
        self.phase_densities[0,1,:] = fprop_block.rho_V * np.ones(ctes.n_volumes).astype(float)

    def inputs_water_properties(self):
        self.phase_densities[0,ctes.n_phases-1,:] = data_loaded['compositional_data']['water_data']['rho_W']
        self.ksi_W0 = self.phase_densities[0,ctes.n_phases-1,:] / ctes.Mw_w
        self.phase_molar_densities[0,ctes.n_phases-1,:] = self.ksi_W0
        self.component_molar_fractions[ctes.n_components-1,ctes.n_phases-1,:] = 1
        #coef_vc7 = np.array([21.573, 0.015122, -27.6563, 0.070615])
        #ctes.vc[ctes.C7 == 1] =  coef_vc7[0] + coef_vc7[1] * np.mean(ctes.Mw[ctes.C7 == 1]) + \
                            #coef_vc7[2] * np.mean([ctes.SG[ctes.C7 == 1]]) + coef_vc7[3] \
                            #* np.mean(ctes.Mw[ctes.C7 == 1]) * np.mean([ctes.SG[ctes.C7 == 1]])
    def update_fluid_properties(self, fprop_block, i):
        self.component_molar_fractions[0:ctes.Nc,0,i] = fprop_block.x[:,np.newaxis]
        self.component_molar_fractions[0:ctes.Nc,1,i] = fprop_block.y[:,np.newaxis]
        self.L[i] = fprop_block.L
        self.V[i] = fprop_block.V
        self.phase_molar_densities[0,0,i] = fprop_block.ksi_L
        self.phase_molar_densities[0,1,i] = fprop_block.ksi_V
        self.phase_densities[0,0,i] = fprop_block.rho_L
        self.phase_densities[0,1,i] = fprop_block.rho_V
