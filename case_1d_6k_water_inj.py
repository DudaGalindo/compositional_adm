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
        fx = open('xP.txt','r')
        x_CMG = [float(line.rstrip('\n\r')) for line in fx]

        fP = open('P.txt','r')
        P_CMG = [float(line.rstrip('\n\r')) for line in fP]

        fSw = open('Sw.txt','r')
        Sw_CMG = [float(line.rstrip('\n\r')) for line in fSw]

        fSo = open('So.txt','r')
        So_CMG = [float(line.rstrip('\n\r')) for line in fSo]

        datas = np.load('flying/results_water_inj_6k_case_6979.npy', allow_pickle=True)
        import pdb; pdb.set_trace()
        for data in datas[2:]:
            Sw = data[5]
            So = data[6]
            Oil_p = data[8]
            Gas_p = data[9]
            pressure = data[4]/1e3
            time = data[3]
            x = np.linspace(0.619614,2723.01,500)

        plt.figure(1)
        plt.title('t = 200 days')
        plt.plot(x, pressure, 'k', x_CMG, P_CMG, 'r')
        plt.ylabel('Pressure (MPa)')
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
        plt.savefig('results/compositional/saturation_water_6k_inj' + '.png')

        import pdb; pdb.set_trace()
