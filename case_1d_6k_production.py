import numpy as np
import matplotlib.pyplot as plt
import os
import math

flying = 'flying'
name = 'results'
arquivos = os.listdir(flying)

for arq in arquivos:
    if  arq.startswith(name):
        p_CMG = np.array([10.34, 10.34, 10.34, 10.3399, 10.3398, 10.3394, 10.3387, 10.3367, 10.332, 10.3228,
                            10.2978, 10.2869, 10.2505, 10.2063, 10.1167, 10.0018, 9.92096, 9.65452, 9.50886, 9.0008,
                            8.9603])
        x_CMG  = np.array([0.0, 243.07, 521.659, 843.941, 1002.35, 1155.3, 1308.24, 1477.58, 1641.45, 1783.47,
                            1952.81, 1996.51, 2094.83, 2176.77, 2280.55, 2367.95, 2406.19, 2526.36, 2575.52, 2723.01,
                            2725.74])

        sat_CMG = np.array([0.619614, 0.619614, 0.61961, 0.619608, 0.619594, 0.61951, 0.619363, 0.61903, 0.618223,
                            0.617101, 0.614912, 0.611742, 0.608856, 0.604673, 0.600107, 0.59602, 0.591843])
        x2_CMG = np.array([0.0, 548.971, 898.565, 996.888, 1209.92, 1510.35, 1674.23, 1816.25, 1996.51, 2111.22,
                            2236.85, 2340.64, 2417.11, 2499.05, 2570.06, 2635.61, 2717.54])
        datas = np.load('flying/results_prod_6k_caset2_3832.npy', allow_pickle=True)
        import pdb; pdb.set_trace()
        for data in datas[2:]:
            saturation = data[6]
            Oil_p = data[8]
            Gas_p = data[9]
            pressure = data[4]/1e6
            time = data[3]
            x = np.linspace(0.619614,2723.01,500)

        plt.figure(1)
        plt.title('t = 200 days')
        plt.plot(x, pressure, 'k', x_CMG, p_CMG, 'r')
        plt.ylabel('Pressure (MPa)')
        plt.xlabel('Distance')
        plt.legend(('PADMEC', 'CMG'))
        plt.grid()
        plt.savefig('results/compositional/pressure_6k_prod' + '.png')

        plt.figure(2)
        plt.title('t = 200 days')
        plt.plot(x, saturation, 'k', x2_CMG, sat_CMG, 'r')
        plt.legend(('PADMEC', 'CMG'))
        plt.ylabel('Oil saturation')
        plt.xlabel('Distance')
        plt.grid()
        plt.savefig('results/compositional/saturation_6k_prod' + '.png')

        import pdb; pdb.set_trace()
