from ..directories import data_loaded
from ..data_class.data_manager import DataManager
from ..utils import relative_permeability2, phase_viscosity, capillary_pressure
from ..utils import constants as ctes
from .. import directories as direc
from .stability_check import StabilityCheck
from . import equation_of_state
import numpy as np


class PropertiesCalc:
    def __init__(self):
        # Ver um jeito melhor de declarar isso de modo que seja só uma vez e juntar com as propriedades iniciais da água pra tirar de fprop
        self.relative_permeability_class = getattr(relative_permeability2, data_loaded['compositional_data']['relative_permeability'])
        self.relative_permeability = self.relative_permeability_class()
        self.phase_viscosity_class = getattr(phase_viscosity, data_loaded['compositional_data']['phase_viscosity'])
        self.EOS_class = getattr(equation_of_state, data_loaded['compositional_data']['equation_of_state'])

    def run_outside_loop(self, M, fprop):
        self.Sw_con = fprop.Sw
        self.update_porous_volume(fprop)
        if ctes.load_w:
            M.data['saturation'] = self.update_water_saturation(fprop, fprop.component_mole_numbers[-1,:])
            fprop.Sw = M.data['saturation']
            fprop.phase_molar_densities, fprop.phase_densities = \
            self.set_water_properties(fprop, fprop.phase_molar_densities, fprop.phase_densities)
        fprop.So, fprop.Sg = self.update_saturations(M.data['saturation'],
                                        fprop.phase_molar_densities, fprop.L, fprop.V)
        self.set_initial_volume(fprop)
        self.set_initial_mole_numbers(fprop)
        fprop.mobilities = self.update_mobilities(fprop, fprop.So, fprop.Sg, fprop.Sw,
                         fprop.phase_molar_densities, fprop.component_molar_fractions)
        self.update_capillary_pressure(fprop)

    def run_inside_loop(self, M, fprop):
        self.update_porous_volume(fprop)
        if ctes.load_w:
            M.data['saturation'] = self.update_water_saturation(fprop, fprop.component_mole_numbers[-1,:])
            fprop.Sw = M.data['saturation']
            fprop.phase_molar_densities, fprop.phase_densities = \
            self.set_water_properties(fprop, fprop.phase_molar_densities, fprop.phase_densities)
        self.update_mole_numbers(fprop)
        self.update_total_volume(fprop)
        fprop.So, fprop.Sg = self.update_saturations(M.data['saturation'],
                                        fprop.phase_molar_densities, fprop.L, fprop.V)
        fprop.mobilities = self.update_mobilities(fprop, fprop.So, fprop.Sg, fprop.Sw,
                         fprop.phase_molar_densities, fprop.component_molar_fractions)
        self.update_capillary_pressure(fprop)

    def set_water_properties(self, fprop, phase_molar_densities, phase_densities):
        phase_molar_densities[0, ctes.n_phases-1,:] = fprop.ksi_W
        phase_densities[0,ctes.n_phases-1,:] = fprop.rho_W
        return phase_molar_densities, phase_densities

    def set_initial_volume(self, fprop):
        self.Vo = fprop.Vp * fprop.So
        self.Vg = fprop.Vp * fprop.Sg
        self.Vw = fprop.Vp * fprop.Sw
        fprop.Vt = self.Vo + self.Vg + self.Vw

    def set_initial_mole_numbers(self, fprop):
        fprop.phase_mole_numbers = np.zeros([1, ctes.n_phases, ctes.n_volumes])

        if ctes.load_k:
            fprop.phase_mole_numbers[0,0,:] = fprop.phase_molar_densities[0,0,:] * self.Vo
            fprop.phase_mole_numbers[0,1,:] = fprop.phase_molar_densities[0,1,:] * self.Vg
        if ctes.load_w:
            fprop.phase_mole_numbers[0,ctes.n_phases-1,:] = fprop.ksi_W * self.Vw

        component_phase_mole_numbers = fprop.component_molar_fractions * fprop.phase_mole_numbers
        fprop.component_mole_numbers = np.sum(component_phase_mole_numbers, axis = 1)

    def update_porous_volume(self, fprop):
        fprop.Vp = ctes.porosity * ctes.Vbulk * (1 + ctes.Cf*(fprop.P - ctes.Pf))

    def update_saturations(self, Sw, phase_molar_densities, L, V):
        if ctes.load_k:
            Sg = (1. - Sw) * \
                (V / phase_molar_densities[0,1,:]) / \
                (V / phase_molar_densities[0,1,:] +
                L / phase_molar_densities[0,0,:])
            So = 1 - Sw - Sg
        else: So = np.zeros(ctes.n_volumes); Sg = np.zeros(ctes.n_volumes)
        return So, Sg

    '''def update_saturations(self, fprop, phase_mole_numbers, phase_molar_densities, Sw):
        if ctes.load_k:
            So = phase_mole_numbers[0,0,:]/phase_molar_densities[0,0,:]/fprop.Vt
            Sg = phase_mole_numbers[0,1,:]/phase_molar_densities[0,1,:]/fprop.Vt
        else: So = np.zeros(ctes.n_volumes) ; Sg = np.zeros(ctes.n_volumes)
        return So, Sg'''

    def update_mole_numbers(self, fprop):
        # este daqui foi criado separado pois, quando compressivel, o volume poroso pode
        #diferir do volume total, servindo como um termo de correção de erro na equação da pressão,
        #como se sabe. A equação do outside loop ela relaciona o volume total com o poroso, de modo
        #que eles nunca vão ser diferentes e um erro vai ser propagado por toda a simulação. Todavia,
        #ele funciona para o primeiro passo de tempo uma vez que a pressão não mudou e Vp = Vt ainda.

        fprop.phase_mole_numbers = np.empty([1, ctes.n_phases, ctes.n_volumes])

        if ctes.load_k:
            fprop.phase_mole_numbers[0,0,:] = np.sum(fprop.component_mole_numbers[0:ctes.Nc,:], axis = 0) * fprop.L
            fprop.phase_mole_numbers[0,1,:] = np.sum(fprop.component_mole_numbers[0:ctes.Nc,:], axis = 0) * fprop.V

        if ctes.load_w:
            fprop.phase_mole_numbers[0,ctes.n_phases-1,:] = fprop.component_mole_numbers[ctes.n_components-1,:]
        else: fprop.phase_mole_numbers[0,ctes.n_phases-1,:] = np.zeros(ctes.n_volumes)

    def update_total_volume(self, fprop):
        fprop.Vt = np.sum(fprop.phase_mole_numbers / fprop.phase_molar_densities, axis = 1).ravel()

    def update_relative_permeabilities(self, fprop, So, Sg, Sw):
        Sgr = float(direc.data_loaded['compositional_data']['residual_saturations']['Sgr'])
        Swr = float(direc.data_loaded['compositional_data']['residual_saturations']['Swr'])

        saturations = np.array([So, Sg, Sw])
        kro,krg,krw, Sor = self.relative_permeability(fprop, saturations)
        relative_permeabilities = np.zeros([1, ctes.n_phases, ctes.n_volumes])
        if ctes.load_k:
            relative_permeabilities[0,0,:] = kro
            relative_permeabilities[0,1,:] = krg
        if ctes.load_w:
            relative_permeabilities[0, ctes.n_phases-1,:] = krw

        return relative_permeabilities

    def update_phase_viscosities(self, fprop, phase_molar_densities, component_molar_fractions):
        phase_viscosities = np.empty([1, ctes.n_phases, ctes.n_volumes])
        if ctes.load_k:
            phase_viscosity = self.phase_viscosity_class(fprop, phase_molar_densities)
            #phase_viscosities[0,0:2,:] = 0.02*np.ones([2,ctes.n_volumes]) #only for BL test
            #phase_viscosities[0,0:2,:] = 0.001*np.ones([2,ctes.n_volumes]) #only for Dietz test
            phase_viscosities[0,0:2,:] = phase_viscosity(fprop, component_molar_fractions)
        if ctes.load_w:
            phase_viscosities[0,ctes.n_phases-1,:] = data_loaded['compositional_data']['water_data']['mi_W']
        return phase_viscosities

    def update_mobilities(self, fprop, So, Sg, Sw, phase_molar_densities, component_molar_fractions):
        krs = self.update_relative_permeabilities(fprop, So, Sg, Sw)
        mis = self.update_phase_viscosities(fprop, phase_molar_densities, component_molar_fractions)
        mobilities = krs / mis
        return mobilities

    def update_capillary_pressure(self, fprop):
        """ not working yet"""
        #get_capillary_pressure = getattr(capillary_pressure, data_loaded['compositional_data']['capillary_pressure'])
        #get_capillary_pressure = get_capillary_pressure(data_loaded, data_impress, fprop.phase_molar_densities, fprop.component_molar_fractions)
        #Pcow, Pcog = get_capillary_pressure(data_loaded, fprop.Sw, fprop.So, fprop.Sg)

        fprop.Pcap = np.zeros([ctes.n_phases,ctes.n_volumes])
        # Pcap[0,0,:] = Pcog
        # Pcap[0,1,:] = Pcow

    def update_water_saturation(self, fprop, Nw):
        Swr = float(direc.data_loaded['compositional_data']['residual_saturations']['Swr'])
        fprop.ksi_W = fprop.ksi_W0 * (1 + ctes.Cw * (fprop.P - ctes.Pw))
        fprop.rho_W = fprop.ksi_W * ctes.Mw_w
        Sw = Nw * (1 / fprop.ksi_W) / fprop.Vp #fprop.Vp #or Vt ?
        #Sw[Nk[ctes.n_components-1,:] <= 1.000001*self.Nw] = self.Sw_con[Nk[ctes.n_components-1,:] <= 1.000001*self.Nw]
        Sw[Sw<self.Sw_con] = self.Sw_con[Sw<self.Sw_con]
        return Sw
