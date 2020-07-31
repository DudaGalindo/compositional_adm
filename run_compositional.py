import numpy as np
from packs.directories import data_loaded
from packs import directories as direc
from packs.running.compositional_initial_mesh_properties import initial_mesh
from packs.compositional.compositionalIMPEC import CompositionalFVM
from packs.compositional.stability_check import StabilityCheck
from packs.compositional.properties_calculation import PropertiesCalc
from packs.compositional.update_time import delta_time
from update_inputs_compositional import FluidProperties
import update_inputs_compositional
from packs.utils import constants as ctes
import os
import time
import cProfile

class run_simulation:
    def __init__(self, name_current, name_all):
        self.name_current_compositional_results =os.path.join(direc.flying, name_current + '.npy')
        self.name_all_compositional_results = os.path.join(direc.flying, name_all)
        self.loop = 0
        self.vpi = 0.0
        self.t = 0.0
        self.oil_production = 0.
        self.gas_production = 0.
        self.use_vpi = data_loaded['use_vpi']
        self.vpi_save = data_loaded['compositional_data']['vpis_para_gravar_vtk']
        self.time_save = np.array(data_loaded['compositional_data']['time_to_save'])
        self.delta_t = data_loaded['compositional_data']['time_data']['delta_t_ini']
        self.mesh_name =  'compositional_'
        self.all_compositional_results = self.get_empty_current_compositional_results()
        self.p1 = PropertiesCalc()

    def initialize(self, load, convert, mesh):
        M, elements_lv0, data_impress, wells = initial_mesh(mesh, load=load, convert=convert)
        ctes.init(M)
        ctes.component_properties()
        fprop = self.get_initial_properties(M, wells, load, data_loaded)
        return M, data_impress, wells, fprop, load

    def get_initial_properties(self, M, wells, load, data_loaded):
        fprop = FluidProperties(wells)
        if ctes.load_k:
            fprop.L, fprop.V, fprop.component_molar_fractions[0:ctes.Nc, 0, :], \
            fprop.component_molar_fractions[0:ctes.Nc, 1, :], fprop.phase_molar_densities[:,0,:], \
            fprop.phase_molar_densities[:,1,:], fprop.phase_densities[:,0,:], fprop.phase_densities[:,1,:]  =  \
            StabilityCheck(fprop, fprop.P).run(fprop, fprop.P, fprop.L, fprop.V, fprop.z)
            #fprop.inputs_fluid_properties()
            self.p2 = StabilityCheck(fprop, fprop.P)
        else: fprop.x = []; fprop.y = []
        if ctes.load_w: fprop.inputs_water_properties(M)

        self.p1.run_outside_loop(M, fprop)
        return fprop

    def run(self, M, wells, fprop, load):
        t0 = time.time()
        t_obj = delta_time(fprop) #get wanted properties in t=n

        self.delta_t = CompositionalFVM().runIMPEC(M, wells, fprop, self.delta_t)

        self.t += self.delta_t

        if ctes.load_k and ctes.compressible_k:
            fprop.L, fprop.V, fprop.component_molar_fractions[0:ctes.Nc, 0, :], \
            fprop.component_molar_fractions[0:ctes.Nc, 1, :], fprop.phase_molar_densities[:,0,:], \
            fprop.phase_molar_densities[:,1,:], fprop.phase_densities[:,0,:], fprop.phase_densities[:,1,:]  =  \
            self.p2.run(fprop, fprop.P, fprop.L, fprop.V, fprop.z)
        self.p1.run_inside_loop(M, fprop)
        #if self.t > 32405: import pdb; pdb.set_trace()
        self.update_vpi(fprop, wells)
        self.delta_t = t_obj.update_delta_t(self.delta_t, fprop, ctes.load_k, self.loop)#get delta_t with properties in t=n and t=n+1
        if len(wells['ws_p'])>0:self.update_production(fprop, wells)
        self.update_loop()
        t1 = time.time()
        dt = t1 - t0
        # Talvez isso esteja antes de self.all_compositional_results dentro de update_current_compositional_results
        if self.use_vpi:
            if np.round(self.vpi,3) in self.vpi_save:
                self.update_current_compositional_results(M, wells, fprop, dt) #ver quem vou salvar
        else:
            if self.time_save[0] == 0.0 or self.t in self.time_save:
                self.update_current_compositional_results(M, wells, fprop, dt)

    def update_loop(self):
        self.loop += 1

    def update_vpi(self, fprop, wells):
        if len(wells['ws_inj'])>0:
            flux_vols_total = wells['values_q_vol']
            flux_total_inj = np.absolute(flux_vols_total)
        else: flux_total_inj = np.zeros(2)

        self.vpi = self.vpi + (flux_total_inj.sum())/sum(fprop.Vp)*self.delta_t

    def get_empty_current_compositional_results(self):

        return [np.array(['loop', 'vpi [s]', 'simulation_time [s]', 't [s]', 'pressure [Pa]', 'Sw', 'So', 'Sg',
                        'Oil_p', 'Gas_p', 'centroids'])]

    def update_production(self, fprop, wells):
        self.oil_production +=  abs(fprop.q_phase[:,0,:].sum()) *self.delta_t
        #abs(sum(np.sum(fprop.q[:,wells['ws_prod']], axis = 0) * self.delta_t * fprop.L[wells['ws_prod']] / \
                                #fprop.phase_molar_densities[:,0,wells['ws_prod']]))
        self.gas_production +=  abs(fprop.q_phase[:,1,:].sum())*self.delta_t 

        #abs(sum(np.sum(fprop.q[:,wells['ws_prod']], axis = 0) * self.delta_t * fprop.V[wells['ws_prod']]  / \
                                #fprop.phase_molar_densities[:,1,wells['ws_prod']]))

    def update_current_compositional_results(self, M, wells, fprop, simulation_time: float = 0.0):

        #total_flux_internal_faces = fprop.total_flux_internal_faces.ravel() #* M.faces.normal[M.faces.internal]
        #total_flux_internal_faces_vector = fprop.total_flux_internal_faces.T * np.abs(M.faces.normal[M.faces.internal])

        self.current_compositional_results = np.array([self.loop, self.vpi, simulation_time,
                    self.t, fprop.P, fprop.Sw, fprop.So, fprop.Sg, self.oil_production,
                    self.gas_production, M.data['centroid_volumes']])
        self.all_compositional_results.append(self.current_compositional_results)

    def export_current_compositional_results(self):
         np.save(self.name_current_compositional_results, self.current_compositional_results)

    def export_all_compositional_results(self):
         np.save(self.name_all_compositional_results + str(self.loop) + '.npy', np.array(self.all_compositional_results))
         self.all_compositional_results = self.get_empty_current_compositional_results()

    def save_infos(self, data_impress, M):
         self.export_current_compositional_results()
         self.export_all_compositional_results()
         data_impress.update_variables_to_mesh()
         data_impress.export_all_datas_to_npz()
         M.core.print(file=self.mesh_name, extension='.h5m', config_input="input_cards/print_settings.yml")
         # M.core.print(file=self.mesh_name, extension='.vtk', config_input="input_cards/print_settings.yml")
