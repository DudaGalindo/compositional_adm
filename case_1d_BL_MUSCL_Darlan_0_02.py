import numpy as np
import matplotlib.pyplot as plt
import os
import math
from scipy.interpolate import interp1d

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
Sw = np.linspace(Swr, 1-Sor, 100000)
td = 0.5

S = (Sw - Swr)/(1 - Swr - Sor)
krw = krw_end*S**nw
kro = kro_end*(1-S)**no
fw = (krw*mi_o) / ((krw*mi_o)+(kro*mi_w))

p1 = np.polyfit(Sw, fw, 12)
p2 = np.polyder(p1)
diff_fw = np.polyval(p2, Sw)

## Find the saturation shock:
for n in range(1, 100000):
    if abs((fw[n])/(Sw[n]-Swr) - diff_fw[n]) < 0.009:
        Swf = Sw[n]
        slope_Swf = (fw[n])/(Sw[n]-Swr)
        a = n

##Water saturation profile of water front:
Sw2 = np.linspace(Swr, Swf, 5)
xD2 = np.linspace(slope_Swf*td, slope_Swf*td, 5)

## Water saturation profile before water front:
Sw1 = np.zeros(a-1)
for i in range(0,a-1):
    Sw1[i] = Swr
    xD1 = np.linspace(1, slope_Swf*td,a-1)

## Water saturation profile behind water front:
Sw3 = np.zeros(100000-a)
xD3 = np.zeros(100000-a)
for j in range(a,100000):
    Sw3[j-a] = Sw[j]
    xD3[j-a] = diff_fw[j]*td

#Sw3[xD3<0] = Sw3[xD3<0][0]
#xD3[xD3<0] = 0
xD = np.append(xD1,xD2)
xD = np.append(xD,xD3)
SwD = np.append(Sw1,Sw2)
SwD = np.append(SwD,Sw3)
f = interp1d(xD,SwD)

for  arq in arquivos:
    if  arq.startswith(name):
        '''-------------------------MUSCL LLF RESULTS------------------------'''
        datas = np.load('flying/results_BL_Darlan_8t_875.npy', allow_pickle=True)

        for data in datas[datas.shape[0]-1:]:
            Sw8 = data[5]
            x8 = np.linspace(x[-1],1,8)
            n = 8
            e8_L1_MUSCL = (sum(abs(f(x8)-Sw8))*(1/n))

        datas = np.load('flying/results_BL_Darlan_16t_1658.npy', allow_pickle=True)

        for data in datas[datas.shape[0]-1:]:
            Sw16 = data[5]
            x16 = np.linspace(x[-1],1,16)
            n = 16
            e16_L1_MUSCL = (sum(abs(f(x16)-Sw16))*(1/n))

        datas = np.load('flying/results_BL_Darlan_32t_3213.npy', allow_pickle=True)

        for data in datas[datas.shape[0]-1:]:
            Sw32 = data[5]
            x32 = np.linspace(x[-1],1,32)
            n = 32
            e32_L1_MUSCL = (sum(abs(f(x32)-Sw32))*(1/n))

        datas = np.load('flying/results_BL_Darlan_64t_1255.npy', allow_pickle=True)

        for data in datas[datas.shape[0]-1:]:
            Sw64 = data[5]
            x64 = np.linspace(xD[-1],1,64)
            n = 64
            e64_L1_MUSCL = (sum(abs(f(x64)-Sw64))*(1/n))

        datas = np.load('flying/results_BL_Darlan_128t_676.npy', allow_pickle=True)

        for data in datas[datas.shape[0]-1:]:
            Sw128 = data[5]
            x128 = np.linspace(xD[-1],1,128)
            n = 128
            e128_L1_MUSCL = (sum(abs(f(x128)-Sw128))*(1/n))

        datas = np.load('flying/results_BL_Darlan_256t_1351.npy', allow_pickle=True)

        for data in datas[datas.shape[0]-1:]:
            Sw256 = data[5]
            x256 = np.linspace(xD[-1],1,256)
            n = 256
            e256_L1_MUSCL = (sum(abs(f(x256)-Sw256))*(1/n))

        datas = np.load('flying/results_BL_Darlan_512t_2701.npy', allow_pickle=True)

        for data in datas[datas.shape[0]-1:]:
            Sw512 = data[5]
            x512 = np.linspace(xD[-1],1,512)
            n = 512
            e512_L1_MUSCL = (sum(abs(f(x512)-Sw512))*(1/n))

        '''----------------------------UPWIND RESULTS------------------------'''

        datas = np.load('flying/results_BL_Darlan_8_upw_779.npy', allow_pickle=True)

        for data in datas[datas.shape[0]-1:]:
            Sw8_upw = data[5]
            n = 8
            e8_L1_upw = (sum(abs(f(x8)-Sw8_upw))*(1/n))

        datas = np.load('flying/results_BL_Darlan_16_upw_1395.npy', allow_pickle=True)

        for data in datas[datas.shape[0]-1:]:
            Sw16_upw = data[5]
            x16_upw = np.linspace(0,1,16)
            n = 16
            e16_L1_upw = (sum(abs(f(x16)-Sw16_upw))*(1/n))

        datas = np.load('flying/results_BL_Darlan_32_upw_2588.npy', allow_pickle=True)

        for data in datas[datas.shape[0]-1:]:
            Sw32_upw = data[5]
            x32_upw = np.linspace(0,1,32)
            n = 32
            e32_L1_upw = sum(abs(f(x32)-Sw32_upw))*(1/n)

        datas = np.load('flying/results_BL_Darlan_64_upw_991.npy', allow_pickle=True)

        for data in datas[datas.shape[0]-1:]:
            Sw64_upw = data[5]
            x64_upw = np.linspace(0,1,64)
            n = 64
            e64_L1_upw = sum(abs(f(x64)-Sw64_upw))*(1/n)

        datas = np.load('flying/results_BL_Darlan_128_upw_477.npy', allow_pickle=True)

        for data in datas[datas.shape[0]-1:]:
            Sw128_upw = data[5]
            x128_upw = np.linspace(0,1,128)
            n = 128
            e128_L1_upw = sum(abs(f(x128)-Sw128_upw))*(1/n)

        datas = np.load('flying/results_BL_Darlan_256_upw_532.npy', allow_pickle=True)

        for data in datas[datas.shape[0]-1:]:
            Sw256_upw = data[5]
            x256_upw = np.linspace(0,1,256)
            n = 256
            e256_L1_upw = (sum(abs(f(x256)-Sw256_upw))*(1/n))

        datas = np.load('flying/results_BL_Darlan_512_upw_1057.npy', allow_pickle=True)

        for data in datas[datas.shape[0]-1:]:
            Sw512_upw = data[5]
            x512_upw = np.linspace(0,1,512)
            n = 512
            e512_L1_upw = (sum(abs(f(x512)-Sw512_upw))*(1/n))



        plt.figure(30)
        x = np.log10(np.array([8,16,32,64,128,256, 512]))
        #y_FR = np.log10(np.array([e8_L1_FR2, e16_L1_FR2, e32_L1_FR2, e64_L1_FR2, e128_L1_FR2, e256_L1_FR2, e512_L1_FR2]))
        y_MUSCL = np.log10(np.array([e8_L1_MUSCL, e16_L1_MUSCL, e32_L1_MUSCL, e64_L1_MUSCL, e128_L1_MUSCL, e256_L1_MUSCL, e512_L1_MUSCL]))
        y_upw = np.log10(np.array([e8_L1_upw, e16_L1_upw, e32_L1_upw, e64_L1_upw, e128_L1_upw, e256_L1_upw, e512_L1_upw]))

        y_ref = -x-0.1
        plt.plot(x, y_MUSCL, 'g', x, y_ref, 'b', x, y_upw, 'y')
        plt.title('Convergence rate - L1 norm')
        plt.ylabel('$log_{10}({E}_{L_1})$')
        plt.xlabel('$log_{10}(N)$')
        plt.legend(('MUSCL-2nd order','Reference Line', 'FOU'))
        plt.grid()
        plt.savefig('results/compositional/FR/BL_Darlan_L1_convergence_order_teste' +'.png')
        import pdb; pdb.set_trace()
