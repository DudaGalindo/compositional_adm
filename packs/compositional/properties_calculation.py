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
        if ctes.load_w:
            self.set_water_properties(fprop)
        self.update_porous_volume(fprop)
        self.update_saturations(M, fprop)
        self.set_initial_volume(fprop)
        self.set_initial_mole_numbers(fprop)
        self.run_all(fprop)

    def run_inside_loop(self, M, fprop):
        if ctes.load_w:
            self.update_water_saturation(M, fprop)
            self.set_water_properties(fprop)
        self.update_porous_volume( fprop)
        self.update_saturations(M, fprop)
        self.update_mole_numbers(fprop)
        self.update_total_volume(fprop)
        self.run_all(fprop)

    def run_all(self, fprop):
        self.update_mobilities(fprop)
        self.update_capillary_pressure(fprop)

    def set_water_properties(self, fprop):
        fprop.phase_molar_densities[0, ctes.n_phases-1,:] = fprop.ksi_W
        fprop.phase_densities[0,ctes.n_phases-1,:] = fprop.rho_W

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

    def update_saturations(self, M, fprop):
        fprop.Sw = M.data['saturation']
        if ctes.load_k:
            fprop.Sg = (1. - fprop.Sw) * \
                (fprop.V / fprop.phase_molar_densities[0,1,:]) / \
                (fprop.V / fprop.phase_molar_densities[0,1,:] +
                fprop.L / fprop.phase_molar_densities[0,0,:] )
            #fprop.Sg[fprop.Sg < 0] = 0
            fprop.So = 1 - fprop.Sw - fprop.Sg
        else: fprop.So = np.zeros(len(fprop.Sw)); fprop.Sg = np.zeros(len(fprop.Sw))

    def update_mole_numbers(self, fprop):
        # este daqui foi criado separado pois, quando compressivel, o volume poroso pode
        #diferir do volume total, servindo como um termo de correção de erro na equação da pressão,
        #como se sabe. A equação do outside loop ela relaciona o volume total com o poroso, de modo
        #que eles nunca vão ser diferentes e um erro vai ser propagado por toda a simulação. Todavia,
        #ele funciona para o primeiro passo de tempo uma vez que a pressão não mudou e Vp = Vt ainda.

        fprop.phase_mole_numbers = np.zeros([1, ctes.n_phases, ctes.n_volumes])

        if ctes.load_k:
            #fprop.L[fprop.L>1] = 1
            fprop.phase_mole_numbers[0,0,:] = np.sum(fprop.component_mole_numbers[0:ctes.Nc,:], axis = 0) * \
                                               fprop.L
            fprop.phase_mole_numbers[0,1,:] = np.sum(fprop.component_mole_numbers[0:ctes.Nc,:], axis = 0) * \
                                               fprop.V
            #fprop.phase_mole_numbers[0,1,fprop.L!=0] = fprop.phase_mole_numbers[0,0,fprop.L!=0]*fprop.V[fprop.L!=0]/fprop.L[fprop.L!=0]
            #fprop.phase_mole_numbers[0,1,fprop.L==0] = fprop.phase_mole_numbers[0,0,fprop.L==0]
            #fprop.phase_mole_numbers[0,1,:] = fprop.component_mole_numbers[fprop.y!=0,:][0,:]/fprop.y[fprop.y!=0][0]

        if ctes.load_w:
            fprop.phase_mole_numbers[0,ctes.n_phases-1,:] = fprop.component_mole_numbers[ctes.n_components-1,:]
        else: fprop.mole_number_w = np.zeros(ctes.n_volumes)

    def update_total_volume(self, fprop):
        fprop.Vt = np.sum(fprop.phase_mole_numbers / fprop.phase_molar_densities, axis = 1).ravel()

    def update_relative_permeabilities(self, fprop):
        Sgr = float(direc.data_loaded['compositional_data']['residual_saturations']['Sgr'])
        Swr = float(direc.data_loaded['compositional_data']['residual_saturations']['Swr'])

        saturations = np.array([fprop.So, fprop.Sg, fprop.Sw])
        kro,krg,krw, Sor = self.relative_permeability(saturations)
        relative_permeabilities = np.zeros([1, ctes.n_phases, ctes.n_volumes])
        if ctes.load_k:
            relative_permeabilities[0,0,:] = kro
            relative_permeabilities[0,1,:] = krg
        if ctes.load_w:
            relative_permeabilities[0, ctes.n_phases-1,:] = krw
            #if ctes.load_k:
                #if any(fprop.Sw > (1 - Sor - Sgr)) or any(fprop.So > (1 - Swr - Sgr)):
                #    raise ValueError('valor errado da saturacao - mudar delta_t_ini')
        return relative_permeabilities

    def update_phase_viscosities(self, fprop):
        phase_viscosities = np.empty([1, ctes.n_phases, ctes.n_volumes])
        if ctes.load_k:
            phase_viscosity = self.phase_viscosity_class(fprop)
            #phase_viscosities[0,0:2,:] = 0.02*np.ones([2,ctes.n_volumes]) #only for BL test
            #phase_viscosities[0,0:2,:] = 0.001*np.ones([2,ctes.n_volumes]) #only for Dietz test
            #phase_viscosities[0,0:2,:] = 0.000249*np.ones([2,ctes.n_volumes]) #only for Dietz test
            phase_viscosities[0,0:2,:] = phase_viscosity(fprop)
        if ctes.load_w:
            phase_viscosities[0,ctes.n_phases-1,:] = data_loaded['compositional_data']['water_data']['mi_W']
        return phase_viscosities

    def update_mobilities(self, fprop):
        krs = self.update_relative_permeabilities(fprop)
        mis = self.update_phase_viscosities(fprop)
        fprop.mobilities = krs / mis

    def update_capillary_pressure(self, fprop):
        """ not working yet"""
        #get_capillary_pressure = getattr(capillary_pressure, data_loaded['compositional_data']['capillary_pressure'])
        #get_capillary_pressure = get_capillary_pressure(data_loaded, data_impress, fprop.phase_molar_densities, fprop.component_molar_fractions)
        #Pcow, Pcog = get_capillary_pressure(data_loaded, fprop.Sw, fprop.So, fprop.Sg)

        fprop.Pcap = np.zeros([ctes.n_phases,ctes.n_volumes])
        # Pcap[0,0,:] = Pcog
        # Pcap[0,1,:] = Pcow

    def update_water_saturation(self, M, fprop):
        Swr = float(direc.data_loaded['compositional_data']['residual_saturations']['Swr'])
        fprop.ksi_W = fprop.ksi_W0 * (1 + ctes.Cw * (fprop.P - ctes.Pw))
        fprop.rho_W = fprop.ksi_W * ctes.Mw_w
        Nw = fprop.ksi_W * ctes.Vbulk * ctes.porosity  #numero de mols de água original
        #import pdb; pdb.set_trace()
        Sw = fprop.component_mole_numbers[ctes.n_components-1,:]\
                        * (1 / fprop.ksi_W) / fprop.Vp #or Vt ?
        M.data['saturation'][fprop.component_mole_numbers[ctes.n_components-1,:]>Nw] = Sw[fprop.component_mole_numbers[ctes.n_components-1,:]>Nw]
