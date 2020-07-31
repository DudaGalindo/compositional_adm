import numpy as np
import matplotlib.pyplot as plt
import os
import math

flying = 'flying'
name = 'results'
arquivos = os.listdir(flying)

for arq in arquivos:
    if  arq.startswith(name):
        p_CMG = np.array([10.340, 10.340,    10.33999219, 10.33992285, 10.33977539, 10.33942773, 10.33846777, 10.33585547,
                          10.32911231, 10.3155625, 10.27865625, 10.26291895, 10.21321191, 10.15170606,  10.03093652,
                          9.894251953, 9.821735352,   9.557222656, 9.424076172, 9.198925781, 8.9601])
        x_CMG  = np.array([0.0,   240.3455,    518.928,     841.21,      1005.082,    1152.566,    1310.976,   1474.8480,
                          1638.72,     1780.7424,    1950.076782, 1993.776,    2092.099121, 2174.035156,  2283.28303,
                          2370.681641, 2408.91846,    2523.6289,   2572.79,     2649.263916, 2725.74])

        sat_CMG = np.array([0.619670331, 0.619670093, 0.619667947, 0.619665861, 0.619653165, 0.619563103, 0.619381785,
                            0.619028687, 0.618038774, 0.616850257, 0.614482999, 0.611638129, 0.60893774, 0.606933534,
                            0.605533183, 0.602042735, 0.59927071, 0.596567392, 0.59623915])
        x2_CMG = np.array([0.0,           546.2399,   895.8336,     994.1567,  1207.19043,  1507.622315, 1676.956787,
                           1818.979248, 1993.776001, 2103.023926, 2234.121582, 2337.907227,  2414.380859, 2463.54242,
                           2496.3169,    2572.790527, 2632.876953, 2714.812744, 2725.74])
        datas = np.load('flying/results_prod_6k_case_204.npy', allow_pickle=True)
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
