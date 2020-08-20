import numpy as np
import matplotlib.pyplot as plt
import os
import math

flying = 'flying'
name = 'results'
arquivos = os.listdir(flying)
krw_end = 1
kro_end = 1
nw = 2
no = 2
Swr = 0.1
Sor = 0.1
mi_w = 1
mi_o = 1
Sw = np.linspace(Swr, 1-Sor, 500)
td = 0.5

S = (Sw - Swr)/(1 - Swr - Sor)
krw = krw_end*S**nw
kro = kro_end*(1-S)**no
fw = (krw*mi_o) / ((krw*mi_o)+(kro*mi_w))

p1 = np.polyfit(Sw, fw, 12)
p2 = np.polyder(p1)
diff_fw = np.polyval(p2, Sw)

## Find the saturation shock:
for n in range(1, 500):
    if abs((fw[n])/(Sw[n]-Swr) - diff_fw[n]) < 0.009:
        Swf = Sw[n]
        slope_Swf = (fw[n])/(Sw[n]-Swr)
        a = n

##Water saturation profile of water front:
Sw2 = np.linspace(Swr, Swf, 5)
xD2 = np.linspace(slope_Swf*td,slope_Swf*td,5)

## Water saturation profile before water front:
Sw1 = np.zeros(a-1)
for i in range(0,a-1):
    Sw1[i] = Swr
    xD1 = np.linspace(1, slope_Swf*td,a-1)

## Water saturation profile behind water front:
Sw3 = np.zeros(500-a+1)
xD3 = np.zeros(500-a+1)
for j in range(a-1,500):
    Sw3[j-(a-1)] = Sw[j]
    xD3[j-(a-1)] = diff_fw[j]*td

xD = np.append(xD1,xD2)
xD = np.append(xD,xD3)
SwD = np.append(Sw1,Sw2)
SwD = np.append(SwD,Sw3)

for  arq in arquivos:
    if  arq.startswith(name):
        '''-------------------------MUSCL LLF RESULTS------------------------'''
        datas = np.load('flying/results_BL_Darlan_8t_875.npy', allow_pickle=True)

        for data in datas[datas.shape[0]-1:]:
            Sw8 = data[5]
            x8 = np.linspace(0,1,8)

        datas = np.load('flying/results_BL_Darlan_16t_1658.npy', allow_pickle=True)

        for data in datas[datas.shape[0]-1:]:
            Sw16 = data[5]
            x16 = np.linspace(0,1,16)

        datas = np.load('flying/results_BL_Darlan_32t_3213.npy', allow_pickle=True)

        for data in datas[datas.shape[0]-1:]:
            Sw32 = data[5]
            x32 = np.linspace(0,1,32)

        datas = np.load('flying/results_BL_Darlan_64t_1255.npy', allow_pickle=True)

        for data in datas[datas.shape[0]-1:]:
            Sw64 = data[5]
            x64 = np.linspace(0,1,64)

        datas = np.load('flying/results_BL_Darlan_128t_676.npy', allow_pickle=True)

        for data in datas[datas.shape[0]-1:]:
            Sw128 = data[5]
            x128 = np.linspace(0,1,128)

        datas = np.load('flying/results_BL_Darlan_256t_1351.npy', allow_pickle=True)

        for data in datas[datas.shape[0]-1:]:
            Sw256 = data[5]
            x256 = np.linspace(0,1,256)

        datas = np.load('flying/results_BL_Darlan_512t_2701.npy', allow_pickle=True)

        for data in datas[datas.shape[0]-1:]:
            Sw512 = data[5]
            x512 = np.linspace(0,1,512)

        '''----------------------------UPWIND RESULTS------------------------'''

        datas = np.load('flying/results_BL_Darlan_8_upw_779.npy', allow_pickle=True)

        for data in datas[datas.shape[0]-1:]:
            Sw8_upw = data[5]
            x8_upw = np.linspace(0,1,8)

        datas = np.load('flying/results_BL_Darlan_16_upw_1395.npy', allow_pickle=True)

        for data in datas[datas.shape[0]-1:]:
            Sw16_upw = data[5]
            x16_upw = np.linspace(0,1,16)

        datas = np.load('flying/results_BL_Darlan_32_upw_2588.npy', allow_pickle=True)

        for data in datas[datas.shape[0]-1:]:
            Sw32_upw = data[5]
            x32_upw = np.linspace(0,1,32)

        datas = np.load('flying/results_BL_Darlan_64_upw_991.npy', allow_pickle=True)

        for data in datas[datas.shape[0]-1:]:
            Sw64_upw = data[5]
            x64_upw = np.linspace(0,1,64)

        datas = np.load('flying/results_BL_Darlan_128_upw_477.npy', allow_pickle=True)
        import pdb; pdb.set_trace()
        for data in datas[datas.shape[0]-1:]:
            Sw128_upw = data[5]
            x128_upw = np.linspace(0,1,128)

        datas = np.load('flying/results_BL_Darlan_256_upw_532.npy', allow_pickle=True)

        for data in datas[datas.shape[0]-1:]:
            Sw256_upw = data[5]
            x256_upw = np.linspace(0,1,256)

        datas = np.load('flying/results_BL_Darlan_512_upw_1057.npy', allow_pickle=True)

        for data in datas[datas.shape[0]-1:]:
            Sw512_upw = data[5]
            x512_upw = np.linspace(0,1,512)

        '''-------------------------MUSCL DW RESULTS-------------------------'''
        datas = np.load('flying/results_BL_Darlan_8_DW_885.npy', allow_pickle=True)

        for data in datas[datas.shape[0]-1:]:
            Sw8_DW = data[5]
            x8_DW = np.linspace(0,1,8)

        datas = np.load('flying/results_BL_Darlan_16_DW_1677.npy', allow_pickle=True)

        for data in datas[datas.shape[0]-1:]:
            Sw16_DW = data[5]
            x16_DW = np.linspace(0,1,16)

        datas = np.load('flying/results_BL_Darlan_32_DW_3252.npy', allow_pickle=True)

        for data in datas[datas.shape[0]-1:]:
            Sw32_DW = data[5]
            x32_DW = np.linspace(0,1,32)

        datas = np.load('flying/results_BL_Darlan_64_DW_1271.npy', allow_pickle=True)

        for data in datas[datas.shape[0]-1:]:
            Sw64_DW = data[5]
            x64_DW = np.linspace(0,1,64)

        datas = np.load('flying/results_BL_Darlan_128_DW_662.npy', allow_pickle=True)

        for data in datas[datas.shape[0]-1:]:
            Sw128_DW = data[5]
            x128_DW = np.linspace(0,1,128)

        datas = np.load('flying/results_BL_Darlan_256_DW_1353.npy', allow_pickle=True)

        for data in datas[datas.shape[0]-1:]:
            Sw256_DW = data[5]
            x256_DW = np.linspace(0,1,256)

        datas = np.load('flying/results_BL_Darlan_512_DW_2805.npy', allow_pickle=True)

        for data in datas[datas.shape[0]-1:]:
            Sw512_DW = data[5]
            x512_DW = np.linspace(0,1,512)


        plt.figure(1)
        plt.plot(x32, Sw32, 'r', x64, Sw64, 'b', x128, Sw128, 'k', x256, Sw256, 'g', x512, Sw512, 'm', xD, SwD, 'y')
        plt.grid()
        loop = data[0]
        plt.legend(('32 elements','64 elements', '128 elements', '256 elements', '512 elements', 'Analytical Solution'))
        plt.title('Buckley-Leverett Solution Example')
        plt.ylabel('Water Saturation')
        plt.xlabel('Dimensionless distance')
        plt.savefig('results/compositional/saturation_BL_Darlan_MUSCL_meshes.png')

        plt.figure(2)
        plt.plot(x32, Sw32, 'r', x32_upw, Sw32_upw, 'g', xD, SwD, 'y')
        plt.legend(('MUSCL','FOUM', 'Analytical Solution'))
        plt.title('Buckley-Leverett Solution Example - 32 elements')
        plt.ylabel('Water Saturation')
        plt.xlabel('Dimensionless distance')
        plt.savefig('results/compositional/saturation_BL_Darlan_32_comparison.png')

        plt.figure(3)
        plt.plot(x64, Sw64, 'r', x64_upw, Sw64_upw, 'g', xD, SwD, 'y')
        plt.legend(('MUSCL','FOUM', 'Analytical Solution'))
        plt.title('Buckley-Leverett Solution Example - 64 elements')
        plt.ylabel('Water Saturation')
        plt.xlabel('Dimensionless distance')
        plt.savefig('results/compositional/saturation_BL_Darlan_64_comparison.png')

        plt.figure(4)
        plt.plot(x128, Sw128, 'r', x128_upw, Sw128_upw, 'g', xD, SwD, 'y')
        plt.legend(('MUSCL','FOUM', 'Analytical Solution'))
        plt.title('Buckley-Leverett Solution Example - 128 elements')
        plt.ylabel('Water Saturation')
        plt.xlabel('Dimensionless distance')
        plt.savefig('results/compositional/saturation_BL_Darlan_128_comparison.png')

        plt.figure(5)
        plt.plot(x256, Sw256, 'r', x256_upw, Sw256_upw, 'g', xD, SwD, 'y')
        plt.legend(('MUSCL','FOUM', 'Analytical Solution'))
        plt.title('Buckley-Leverett Solution Example - 256 elements')
        plt.ylabel('Water Saturation')
        plt.xlabel('Dimensionless distance')
        plt.savefig('results/compositional/saturation_BL_Darlan_256_comparison.png')

        plt.figure(6)
        plt.plot(x512, Sw512, 'r', x512_upw, Sw512_upw, 'g', xD, SwD, 'y')
        plt.legend(('MUSCL','FOUM', 'Analytical Solution'))
        plt.title('Buckley-Leverett Solution Example - 512 elements')
        plt.ylabel('Water Saturation')
        plt.xlabel('Dimensionless distance')
        plt.savefig('results/compositional/saturation_BL_Darlan_512_comparison.png')

        plt.figure(7)
        plt.plot(x8, Sw8, 'r', x8_upw, Sw8_upw, 'g', xD, SwD, 'y')
        plt.legend(('MUSCL','FOUM', 'Analytical Solution'))
        plt.title('Buckley-Leverett Solution Example - 8 elements')
        plt.ylabel('Water Saturation')
        plt.xlabel('Dimensionless distance')
        plt.savefig('results/compositional/saturation_BL_Darlan_8_comparison.png')

        plt.figure(8)
        plt.plot(x16, Sw16, 'r', x16_upw, Sw16_upw, 'g', xD, SwD, 'y')
        plt.legend(('MUSCL','FOUM', 'Analytical Solution'))
        plt.title('Buckley-Leverett Solution Example - 16 elements')
        plt.ylabel('Water Saturation')
        plt.xlabel('Dimensionless distance')
        plt.savefig('results/compositional/saturation_BL_Darlan_16_comparison.png')

        '''LLF e DW comparison'''
        plt.figure(9)
        plt.plot(x32, Sw32, 'r', x32_DW, Sw32_DW, 'g', xD, SwD, 'y')
        plt.legend(('MUSCL-LLF','MUSCL-DW', 'Analytical Solution'))
        plt.title('Buckley-Leverett Solution Example - 32 elements')
        plt.ylabel('Water Saturation')
        plt.xlabel('Dimensionless distance')
        plt.savefig('results/compositional/saturation_BL_Darlan_32_comparisonDW.png')

        plt.figure(10)
        plt.plot(x64, Sw64, 'r', x64_DW, Sw64_DW, 'g', xD, SwD, 'y')
        plt.legend(('MUSCL-LLF','FOUM-DW', 'Analytical Solution'))
        plt.title('Buckley-Leverett Solution Example - 64 elements')
        plt.ylabel('Water Saturation')
        plt.xlabel('Dimensionless distance')
        plt.savefig('results/compositional/saturation_BL_Darlan_64_comparisonDW.png')

        plt.figure(11)
        plt.plot(x128, Sw128, 'r', x128_DW, Sw128_DW, 'g', xD, SwD, 'y')
        plt.legend(('MUSCL-LLF','MUSCL-DW', 'Analytical Solution'))
        plt.title('Buckley-Leverett Solution Example - 128 elements')
        plt.ylabel('Water Saturation')
        plt.xlabel('Dimensionless distance')
        plt.savefig('results/compositional/saturation_BL_Darlan_128_comparisonDW.png')

        plt.figure(12)
        plt.plot(x256, Sw256, 'r', x256_DW, Sw256_DW, 'g', xD, SwD, 'y')
        plt.legend(('MUSCL-LLF','MUSCL-DW', 'Analytical Solution'))
        plt.title('Buckley-Leverett Solution Example - 256 elements')
        plt.ylabel('Water Saturation')
        plt.xlabel('Dimensionless distance')
        plt.savefig('results/compositional/saturation_BL_Darlan_256_comparisonDW.png')

        plt.figure(13)
        plt.plot(x512, Sw512, 'r', x512_DW, Sw512_DW, 'g', xD, SwD, 'y')
        plt.legend(('MUSCL-LLF','MUSCL-DW', 'Analytical Solution'))
        plt.title('Buckley-Leverett Solution Example - 512 elements')
        plt.ylabel('Water Saturation')
        plt.xlabel('Dimensionless distance')
        plt.savefig('results/compositional/saturation_BL_Darlan_512_comparisonDW.png')

        plt.figure(14)
        plt.plot(x8, Sw8, 'r', x8_DW, Sw8_DW, 'g', xD, SwD, 'y')
        plt.legend(('MUSCL-LLF','MUSCL-DW', 'Analytical Solution'))
        plt.title('Buckley-Leverett Solution Example - 8 elements')
        plt.ylabel('Water Saturation')
        plt.xlabel('Dimensionless distance')
        plt.savefig('results/compositional/saturation_BL_Darlan_8_comparisonDW.png')

        plt.figure(15)
        plt.plot(x16, Sw16, 'r', x16_DW, Sw16_DW, 'g', xD, SwD, 'y')
        plt.legend(('MUSCL-LLF','MUSCL-DW', 'Analytical Solution'))
        plt.title('Buckley-Leverett Solution Example - 16 elements')
        plt.ylabel('Water Saturation')
        plt.xlabel('Dimensionless distance')
        plt.savefig('results/compositional/saturation_BL_Darlan_16_comparisonDW.png')
        import pdb; pdb.set_trace()
