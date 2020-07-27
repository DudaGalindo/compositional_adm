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
    def __init__(self, wells):
        self.P = np.array([data_loaded['Pressure']['r1']['value']]).astype(float)
        self.T = data_loaded['Temperature']['r1']['value']
        self.component_molar_fractions = np.ones([ctes.n_components, ctes.n_phases, ctes.n_volumes])
        self.phase_molar_densities = np.empty([1, ctes.n_phases, ctes.n_volumes])
        self.phase_densities = np.empty_like(self.phase_molar_densities)
        self.component_mole_numbers = np.empty([ctes.n_components, ctes.n_volumes])
        self.update_initial_porous_volume()
        self.P = self.P * np.ones(ctes.n_volumes)
        self.P[wells['ws_p']] = wells['values_p']
        self.L = np.empty(len(self.P))
        self.V = np.empty(len(self.P))
        if ctes.load_k:
            self.z = np.array([data_loaded['compositional_data']['component_data']['z']]).astype(float).T
            self.z = self.z * np.ones(ctes.n_volumes)

        else: z = []

    def update_initial_porous_volume(self):
        self.Vp = ctes.porosity * ctes.Vbulk * (1 + ctes.Cf*(self.P - ctes.Pf))

    def inputs_fluid_properties(self):
        #self.z = self.z * np.ones(ctes.n_volumes)
        #self.P = self.P * np.ones(ctes.n_volumes)
        self.L = self.L * np.ones(ctes.n_volumes).astype(float)
        self.V = self.V * np.ones(ctes.n_volumes).astype(float)
        #coef_vc7 = np.array([21.573, 0.015122, -27.6563, 0.070615])
        #ctes.vc[ctes.C7 == 1] =  coef_vc7[0] + coef_vc7[1] * np.mean(ctes.Mw[ctes.C7 == 1]) + \
                            #coef_vc7[2] * np.mean([ctes.SG[ctes.C7 == 1]]) + coef_vc7[3] \
                            #* np.mean(ctes.Mw[ctes.C7 == 1]) * np.mean([ctes.SG[ctes.C7 == 1]])

    def inputs_water_properties(self, M):
        self.phase_densities[0,ctes.n_phases-1,:] = data_loaded['compositional_data']['water_data']['rho_W']
        self.ksi_W0 = self.phase_densities[0,ctes.n_phases-1,:] / ctes.Mw_w
        self.ksi_W = self.ksi_W0
        self.rho_W = self.ksi_W * ctes.Mw_w
        self.Sw = M.data['saturation']
        #self.phase_molar_densities[0,ctes.n_phases-1,:] = self.ksi_W0
        self.component_molar_fractions[ctes.n_components-1,ctes.n_phases-1,:] = 1
        self.component_molar_fractions[ctes.n_components-1,0:ctes.n_phases-1,:] = 0
        self.component_molar_fractions[0:ctes.n_components-1,ctes.n_phases-1,:] = 0
        self.component_mole_numbers[-1,:] = self.Vp * self.ksi_W * self.Sw
