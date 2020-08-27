import numpy as np
import matplotlib.pyplot as plt
import os
import math
import csv

flying = 'flying'
name = 'results'
arquivos = os.listdir(flying)

for arq in arquivos:
    if  arq.startswith(name):
        fx = open('x_incomp.txt','r')
        x_CMG = [float(line.rstrip('\n\r')) for line in fx]

        fP = open('P_incomp_low.txt','r')
        P_CMG = [float(line.rstrip('\n\r')) for line in fP]

        fSw = open('Sw_incomp_low.txt','r')
        Sw_CMG = [float(line.rstrip('\n\r')) for line in fSw]

        fSo = open('So_incomp_low.txt','r')
        So_CMG = [float(line.rstrip('\n\r')) for line in fSo]

        fSg = open('Sg_incomp_low.txt','r')
        Sg_CMG = [float(line.rstrip('\n\r')) for line in fSg]
        n=128
        mode = 'LLF'

        datas = np.load('flying/results_water_inj_6k_128_MUSCL_modified_case_2826.npy', allow_pickle=True)
        for data in datas[2:]:
            SwLLF = data[5]
            SoLLF = data[6]
            SgLLF = data[7]
            Oil_p = data[8]
            Gas_p = data[9]
            pressureLLF = data[4]/1e3
            time = data[3]

        datas = np.load('flying/results_water_inj_6k_128_modified_case_2824.npy', allow_pickle=True)
        for data in datas[2:]:
            Sw = data[5]
            So = data[6]
            Sg = data[7]
            Oil_p = data[8]
            Gas_p = data[9]
            pressure = data[4]/1e3
            time = data[3]
            #x1 = np.linspace(0.54624,2725.7376,500)
        datas = np.load('flying/results_water_inj_6k_128_MUSCL_modified_case_upw_2307.npy', allow_pickle=True)
        for data in datas[2:]:
            Sw_MUSCL = data[5]
            So_MUSCL = data[6]
            Sg_MUSCL = data[7]
            Oil_p_MUSCL = data[8]
            Gas_p_MUSCL = data[9]
            pressure_MUSCL = data[4]/1e3
            time_MUSCL = data[3]

        #x = np.linspace(0.54624/n,2731.2/n*(n-1),n)
        x1 = np.linspace(0.54624, 2731.2, n)
        x = x1
        plt.figure(1)
        plt.title('t = 200 days - ' + '{}'.format(n) + 'x1x1 mesh')
        plt.plot(x1, pressureLLF,'g', x1, pressure, 'k', x_CMG, P_CMG, 'r', x, pressure_MUSCL, 'y')
        plt.ylabel('Pressure (kPa)')
        plt.xlabel('Distance')
        plt.legend(('PADMEC-LLF', 'PADMEC-FOUM', 'CMG - 1024 elements', 'PADMEC-MUSCL_UPW'))
        plt.grid()
        plt.savefig('results/compositional/MUSCL_tests/pressure_6k_MUSCL_modified' + '{}'.format(n) + mode +'.png')

        plt.figure(2)
        plt.title('t = 200 days - ' + '{}'.format(n) + 'x1x1 mesh')
        plt.plot(x1, SoLLF, 'g', x1, So, 'k', x_CMG, So_CMG, 'r', x, So_MUSCL, 'y')
        plt.legend(('PADMEC-LLF', 'PADMEC-FOUM', 'CMG - 1024 elements', 'PADMEC-MUSCL_UPW'))
        plt.ylabel('Oil saturation')
        plt.xlabel('Distance')
        plt.grid()
        plt.savefig('results/compositional/MUSCL_tests/saturation_oil_6k_MUSCL_modified' + '{}'.format(n) +mode+ '.png')

        plt.figure(3)
        plt.title('t = 200 days - ' + '{}'.format(n) + 'x1x1 mesh')
        plt.plot(x1, SwLLF, 'g', x1, Sw, 'k', x_CMG, Sw_CMG, 'r', x, Sw_MUSCL, 'y')
        plt.legend(('PADMEC-LLF', 'PADMEC-FOUM', 'CMG - 1024 elements', 'PADMEC-MUSCL_UPW'))
        plt.ylabel('Water saturation')
        plt.xlabel('Distance')
        plt.grid()
        plt.savefig('results/compositional/MUSCL_tests/saturation_water_6k_MUSCL_modified' + '{}'.format(n) + mode+'.png')

        plt.figure(4)
        plt.title('t = 200 days - ' + '{}'.format(n) + 'x1x1 mesh')
        plt.plot(x1, SgLLF, 'g', x1, Sg, 'k', x_CMG, Sg_CMG, 'r', x, Sg_MUSCL, 'y')
        plt.legend(('PADMEC-LLF', 'PADMEC-FOUM', 'CMG', 'PADMEC-MUSCL_UPW'))
        plt.ylabel('Gas saturation')
        plt.xlabel('Distance')
        plt.grid()
        plt.savefig('results/compositional/MUSCL_tests/saturation_gas_6k_MUSCL_modified' +'{}'.format(n) +mode+ '.png')


        import pdb; pdb.set_trace()
        datas = np.load('flying/results_water_inj_6k_8_MUSCL_modified_case_LLF_572.npy', allow_pickle=True)
        for data in datas[2:]:
            Sw_MUSCL8 = data[5]
            So_MUSCL8 = data[6]
            Sg_MUSCL8 = data[7]
            pressure_MUSCL8 = data[4]/1e3
            n=8
            x8 = np.linspace(0.54624/n,2731.2/n*(n-1),n)

        datas = np.load('flying/results_water_inj_6k_16_MUSCL_modified_case_LLF_1020.npy', allow_pickle=True)
        for data in datas[2:]:
            Sw_MUSCL16 = data[5]
            So_MUSCL16 = data[6]
            Sg_MUSCL16 = data[7]
            pressure_MUSCL16 = data[4]/1e3
            n=16
            x16 = np.linspace(0.54624/n,2731.2/n*(n-1),n)

        datas = np.load('flying/results_water_inj_6k_32_MUSCL_modified_case_LLF_1693.npy', allow_pickle=True)
        for data in datas[2:]:
            Sw_MUSCL32 = data[5]
            So_MUSCL32 = data[6]
            Sg_MUSCL32 = data[7]
            pressure_MUSCL32 = data[4]/1e3
            n=32
            x32 = np.linspace(0.54624/n,2731.2/n*(n-1),n)

        datas = np.load('flying/results_water_inj_6k_64_MUSCL_modified_case_LLF_1460.npy', allow_pickle=True)
        for data in datas[2:]:
            Sw_MUSCL64 = data[5]
            So_MUSCL64 = data[6]
            Sg_MUSCL64 = data[7]
            pressure_MUSCL64 = data[4]/1e3
            n=64
            x64 = np.linspace(0.54624/n,2731.2/n*(n-1),n)

        datas = np.load('flying/results_water_inj_6k_128_MUSCL_modified_case_LLF_2826.npy', allow_pickle=True)
        for data in datas[2:]:
            Sw_MUSCL128 = data[5]
            So_MUSCL128 = data[6]
            Sg_MUSCL128 = data[7]
            pressure_MUSCL128 = data[4]/1e3
            n=128
            x128 = np.linspace(0.54624/n,2731.2/n*(n-1),n)



        mode = 'LLF'

        x1 = x
        plt.figure(5)
        plt.title('t = 200 days - ' + '{}'.format(n) + 'x1x1 mesh')
        plt.plot(x8, pressure_MUSCL8, 'r', x16, pressure_MUSCL16,'b',x32, pressure_MUSCL32,'m',
        x64, pressure_MUSCL64,'y', x128, pressure_MUSCL128,'g', x_CMG, P_CMG, 'k',)
        plt.ylabel('Pressure (kPa)')
        plt.xlabel('Distance')
        plt.legend(('8x1x1', '16x1x1', '32x1x1', '64x1x1', '128x1x1', '256x1x1', 'GEM-CMG'))
        plt.grid()
        plt.savefig('results/compositional/MUSCL_tests/pressure_6k_MUSCL_modified_convergence' + mode +'.png')

        plt.figure(6)
        plt.title('t = 200 days - ' + '{}'.format(n) + 'x1x1 mesh')
        plt.plot(x8, So_MUSCL8, 'r', x16, So_MUSCL16,'b',x32, So_MUSCL32,'m',
        x64, So_MUSCL64,'y', x128, So_MUSCL128,'g', x_CMG, So_CMG, 'k',)
        plt.legend(('8x1x1', '16x1x1', '32x1x1', '64x1x1', '128x1x1', '256x1x1', 'GEM-CMG'))
        plt.ylabel('Oil saturation')
        plt.xlabel('Distance')
        plt.grid()
        plt.savefig('results/compositional/MUSCL_tests/saturation_oil_6k_MUSCL_modified_convergence' +mode+ '.png')

        plt.figure(7)
        plt.title('t = 200 days - ' + '{}'.format(n) + 'x1x1 mesh')
        plt.plot(x8, Sw_MUSCL8, 'r', x16, Sw_MUSCL16,'b',x32, Sw_MUSCL32,'m',
        x64, Sw_MUSCL64,'y', x128, Sw_MUSCL128,'g', x_CMG, Sw_CMG, 'k',)
        plt.legend(('8x1x1', '16x1x1', '32x1x1', '64x1x1', '128x1x1', '256x1x1', 'GEM-CMG'))
        plt.ylabel('Water saturation')
        plt.xlabel('Distance')
        plt.grid()
        plt.savefig('results/compositional/MUSCL_tests/saturation_water_6k_MUSCL_modified_convergence' + mode+'.png')

        plt.figure(8)
        plt.title('t = 200 days - ' + '{}'.format(n) + 'x1x1 mesh')
        plt.plot(x8, Sg_MUSCL8, 'r', x16, Sg_MUSCL16,'b',x32, Sg_MUSCL32,'m',
        x64, Sg_MUSCL64,'y', x128, Sg_MUSCL128,'g', x_CMG, Sg_CMG, 'k',)
        plt.legend(('8x1x1', '16x1x1', '32x1x1', '64x1x1', '128x1x1', '256x1x1', 'GEM-CMG'))
        plt.ylabel('Gas saturation')
        plt.xlabel('Distance')
        plt.grid()
        plt.savefig('results/compositional/MUSCL_tests/saturation_gas_6k_MUSCL_modified_convergence' +mode+ '.png')




        '''x = np.linspace(0.54624,2725.7376,500)
        plt.figure(1)
        plt.title('t = 200 days')
        plt.plot(x, pressure, 'k', x_CMG, P_CMG, 'r', x, pressure_MUSCL, 'y')
        plt.ylabel('Pressure (kPa)')
        plt.xlabel('Distance')
        plt.legend(('PADMEC-FOUM', 'CMG', 'PADMEC-MUSCL'))
        plt.grid()
        plt.savefig('results/compositional/pressure_6k_MUSCL_inj' + '.png')

        plt.figure(2)
        plt.title('t = 200 days')
        plt.plot(x, So, 'k', x_CMG, So_CMG, 'r', x, So_MUSCL, 'y')
        plt.legend(('PADMEC-FOUM', 'CMG', 'PADMEC-MUSCL'))
        plt.ylabel('Oil saturation')
        plt.xlabel('Distance')
        plt.grid()
        plt.savefig('results/compositional/saturation_oil_6k_MUSCL_inj' + '.png')

        plt.figure(3)
        plt.title('t = 200 days')
        plt.plot(x, Sw, 'k', x_CMG, Sw_CMG, 'r', x, Sw_MUSCL, 'y')
        plt.legend(('PADMEC-FOUM', 'CMG', 'PADMEC-MUSCL'))
        plt.ylabel('Water saturation')
        plt.xlabel('Distance')
        plt.grid()
        plt.savefig('results/compositional/saturation_water_6k_MUSCL_inj' + '.png')'''
        '''plt.figure(1)
        plt.title('t = 200 days')
        plt.plot(x, pressure, 'k', x_CMG, P_CMG, 'r')
        plt.ylabel('Pressure (kPa)')
        plt.xlabel('Distance')
        plt.legend(('PADMEC', 'CMG'))
        plt.grid()
        plt.savefig('results/compositional/pressure_6k_inj' + '.png')

        plt.figure(2)
        plt.title('t = 200 days')
        plt.plot(x, So, 'k', x_CMG, So_CMG, 'r')
        plt.legend(('PADMEC', 'CMG'))
        plt.ylabel('Oil saturation')
        plt.xlabel('Distance')
        plt.grid()
        plt.savefig('results/compositional/saturation_oil_6k_inj' + '.png')

        plt.figure(3)
        plt.title('t = 200 days')
        plt.plot(x, Sw, 'k', x_CMG, Sw_CMG, 'r')
        plt.legend(('PADMEC', 'CMG'))
        plt.ylabel('Water saturation')
        plt.xlabel('Distance')
        plt.grid()
        plt.savefig('results/compositional/saturation_water_6k_inj' + '.png')'''

        import pdb; pdb.set_trace()
