import numpy as np
import matplotlib.pyplot as plt
import os
import math
from scipy import interpolate

flying = 'flying'
name = 'results_'
arquivos = os.listdir(flying)

""" Analytical Solution """
a = 2000
b = 2000
l = 1000
q = 1000
h = 1

Pi = 2000
Bo = 1.1178
miu = 0.2495
por = 0.2
k = 1.5
cf = 1.04e-5
cr = 5e-4
c = cf+cr
N = 133

Qj = 8.3
Q = -(1/Bo)*Qj/5.614/h

alpha = 157.952*por*c*miu/k
beta = 886.905*Bo*miu/k

t = 365
x = np.linspace(0,2000,N)
y = 840 #840
P = np.zeros(N)

for i in range(0,N):
    d = 0
    for m in range(1,100+1):
        s = 1/(math.pi**2 * (m**2/a**2)) * (1 - np.exp(-math.pi**2/alpha*(m**2/a**2)*t))*np.cos(m*math.pi*l/a)*np.cos(m*math.pi*x[i]/a)
        d = d+s

    f=0
    for n in range(1,100+1):
        j = 1/(math.pi**2 * (n**2/b**2)) * (1 - np.exp(-math.pi**2/alpha*(n**2/b**2)*t))*np.cos(n*math.pi*q/b)*np.cos(n*math.pi*y/b)
        f = f+j

    g=0
    for n in range(1,100+1):
        for m in range(1,100+1):
            z = 1/(math.pi**2*(m**2/a**2+n**2/b**2)) * (1-np.exp(-math.pi**2/alpha*(m**2/a**2+n**2/b**2)*t)) * np.cos(m*math.pi*l/a) * \
            np.cos(n*math.pi*q/b) * np.cos(m*math.pi*x[i]/a) * np.cos(n*math.pi*y/b)
            g = g+z
    P[i] = Pi - beta*Q/(a*b)*(t/alpha+2*d+2*f+4*g)


#    p_resp = np.linspace(0.623843,0,100)
for  arq in arquivos:
    if  arq.startswith(name):

        datas = np.load('flying/results_2d_injection_5_case_452.npy', allow_pickle=True)

        for data in datas[1:]:
            pressure1 = data[4] / 6894.757
            """ Just for the 2D case """
            from packs.utils.utils_old import get_box
            centroids = data[8]
            x1 = np.linspace(0,2000,5)
            p0 = [0,121.9,-0.3048]
            p1 = [609.6,243.84,0.]
            ind_ans = get_box(centroids,np.array([p0,p1]))
            cent_ind = centroids[ind_ans]
            cent_mix = cent_ind[:,0]
            ind_ans_sort = np.argsort(cent_mix)
            pressure1_1 = pressure1[ind_ans]
            pressure1_1 = pressure1_1[ind_ans_sort]

            p0 = [0,243.84,-0.3048]
            p1 = [609.6,365.76,0.]
            ind_ans = get_box(centroids,np.array([p0,p1]))
            cent_ind = centroids[ind_ans]
            cent_mix = cent_ind[:,0]
            ind_ans_sort = np.argsort(cent_mix)
            pressure1_2 = pressure1[ind_ans]
            pressure1_2 = pressure1_2[ind_ans_sort]

            pressure1_ = 0.6*(pressure1_2 - pressure1_1) + pressure1_1
            x1 = np.linspace(0,2000,5)
            #from scipy.interpolate import interp1d
            #f = interp1d(x1, pressure1_)
            #p11 = f(x)

            #tck = interpolate.splrep(x1,pressure1_,s=0)
            #p11 = interpolate.splev(x,tck,der=0)
            e1_max = max(abs(P[0:133:27]-pressure1_))
            e1_L2 = (sum((P[0:133:27]-pressure1_)**2)*(1/5)**2)**(1/2)

            '''p0 = [0, 243.84, -0.3048]
            p1 = [609.6, 365.76, 0.0]
            ind_ans = get_box(centroids,np.array([p0,p1]))
            cent_ind = centroids[ind_ans]
            cent_mix = cent_ind[:,0]
            ind_ans_sort = np.argsort(cent_mix)
            pressure1_ = pressure1[ind_ans]
            pressure1_ = pressure1_[ind_ans_sort]
            #tck = interpolate.splrep(x1,p1_center,s=0)
            #p1_center = interpolate.splev(x,tck,der=0)
            e1center_max = max(abs(P[0:133:27]-pressure1_))
            e1center_L2 = (sum((P[0:133:27]-pressure1_)**2)*(1/5)**2)**(1/2)'''

        datas = np.load('flying/results_2d_injection_15_case_452.npy', allow_pickle=True)

        for data in datas[1:]:
            pressure2 = data[4] / 6894.757
            """ Just for the 2D case """
            from packs.utils.utils_old import get_box
            centroids = data[8]
            x2 = np.linspace(0,2000,15)
            p0 = [0,203.2,-0.3048]
            p1 = [609.6,243.84,0.]
            ind_ans = get_box(centroids,np.array([p0,p1]))
            cent_ind = centroids[ind_ans]
            cent_mix = cent_ind[:,0]
            ind_ans_sort = np.argsort(cent_mix)
            pressure2_1 = pressure2[ind_ans]
            pressure2_1 = pressure2_1[ind_ans_sort]

            p0 = [0,243.84,-0.3048]
            p1 = [609.6,284.48,0.]
            ind_ans = get_box(centroids,np.array([p0,p1]))
            cent_ind = centroids[ind_ans]
            cent_mix = cent_ind[:,0]
            ind_ans_sort = np.argsort(cent_mix)
            pressure2_2 = pressure2[ind_ans]
            pressure2_2 = pressure2_2[ind_ans_sort]

            pressure2_ = 0.8*(pressure2_2 - pressure2_1) + pressure2_1


            #tck = interpolate.splrep(x2,pressure2_,s=0)
            #p2 = interpolate.splev(x,tck,der=0)
            #f = interp1d(x2, pressure2_)
            #p2 = f(x)

            e2_max = max(abs(P[0:133:9]-pressure2_))
            e2_L2 = (sum((P[0:133:9]-pressure2_)**2)*(1/15)**2)**(1/2)
            R2_L2 = math.log(e1_L2/e2_L2,3)
            R2_max = math.log(e1_max/e2_max, 3)

            '''p0 = [0, 284.48, -0.3048]
            p1 = [609.6, 325.12, 0.0]
            ind_ans = get_box(centroids,np.array([p0,p1]))
            cent_ind = centroids[ind_ans]
            cent_mix = cent_ind[:,0]
            ind_ans_sort = np.argsort(cent_mix)
            pressure2_ = pressure2[ind_ans]
            pressure2_ = pressure2_[ind_ans_sort]
            #tck = interpolate.splrep(x2,p2_center,s=0)
            #p2_center = interpolate.splev(x,tck,der=0)
            e2center_max = max(abs(P[0:133:9]-pressure2_))
            e2center_L2 = (sum((P[0:133:9]-pressure2_)**2)*(1/15)**2)**(1/2)'''


        datas = np.load('flying/results_2d_injection_45_case_790.npy', allow_pickle=True)

        for data in datas[1:]:
            pressure3 = data[4] / 6894.757
            """ Just for the 2D case """
            from packs.utils.utils_old import get_box
            centroids = data[8]
            x3 = np.linspace(0,2000,45)
            p0 = [0,243.84,-0.3048]
            p1 = [609.6,257.3866667,0.0]
            ind_ans = get_box(centroids,np.array([p0,p1]))
            cent_ind = centroids[ind_ans]
            cent_mix = cent_ind[:,0]
            ind_ans_sort = np.argsort(cent_mix)
            pressure3_1 = pressure3[ind_ans]
            pressure3_1 = pressure3_1[ind_ans_sort]

            p0 = [0,257.3866667,-0.3048]
            p1 = [609.6,270.9333333,0.0]
            ind_ans = get_box(centroids,np.array([p0,p1]))
            cent_ind = centroids[ind_ans]
            cent_mix = cent_ind[:,0]
            ind_ans_sort = np.argsort(cent_mix)
            pressure3_2 = pressure3[ind_ans]
            pressure3_2 = pressure3_2[ind_ans_sort]

            pressure3_ = 0.42349158*(pressure3_2 - pressure3_1) + pressure3_1

            tck = interpolate.splrep(x3,pressure3_,s=0)
            p3 = interpolate.splev(x,tck,der=0)
            #f = interp1d(x3, pressure3_)
            #p3 = f(x)

            e3_max = max(abs(P[0:133:3]-pressure3_))
            e3_L2 = (sum((P[0:133:3]-pressure3_)**2)*(1/45)**2)**(1/2)
            R3_L2 = math.log(e2_L2/e3_L2,3)
            R3_max = math.log(e2_max/e3_max, 3)

            '''p0 = [0, 298.0266667, -0.3048]
            p1 = [609.6, 311.5733333, 0.0]
            ind_ans = get_box(centroids,np.array([p0,p1]))
            cent_ind = centroids[ind_ans]
            cent_mix = cent_ind[:,0]
            ind_ans_sort = np.argsort(cent_mix)
            pressure3_ = pressure3[ind_ans]
            pressure3_ = pressure3_[ind_ans_sort]
            #tck = interpolate.splrep(x3,p3_center,s=0)
            #p3_center = interpolate.splev(x,tck,der=0)
            e3center_max = max(abs(P[0:133:3]-pressure3_))
            e3center_L2 = (sum((P[0:133:3]-pressure3_)**2)*(1/45)**2)**(1/2)'''


        datas = np.load('flying/results_2d_injection_25_case_452.npy', allow_pickle=True)

        for data in datas[1:]:
            pressure4 = data[4] / 6894.75729
            """ Just for the 2D case """
            from packs.utils.utils_old import get_box
            centroids = data[10]
            p0 = [0,243.84,-0.3048]
            p1 = [609.6,268.224,0.0]
            #p0 = [0., 292.608, -0.3048]
            #p1 = [609.6, 316.992, 0.0]
            ind_ans = get_box(centroids,np.array([p0,p1]))
            cent_ind = centroids[ind_ans]
            cent_mix = cent_ind[:,0]
            ind_ans_sort = np.argsort(cent_mix)
            pressure4_1 = pressure4[ind_ans]
            pressure4_ = pressure4_1[ind_ans_sort]

            x4 = np.linspace(0,2000,25)
            tck = interpolate.splrep(x4,pressure4_,s=0)
            p4 = interpolate.splev(x,tck,der=0)
            e4_max = max(abs(P-p4))
            e4_L2 = (sum((P-p4)**2)*(609.6/25)**2)**(1/2)

        '''datas = np.load('flying/results_2d_injection_case_35__74.npy', allow_pickle=True)

        for data in datas[1:]:
            pressure5 = data[4] / 6894.757
            """ Just for the 2D case """
            from packs.utils.utils_old import get_box
            centroids = data[6]
            p0 = [0,247.3234286,0]
            p1 = [609.6,264.74057,0.3048]
            ind_ans = get_box(centroids,np.array([p0,p1]))
            cent_ind = centroids[ind_ans]
            cent_mix = cent_ind[:,0]
            ind_ans_sort = np.argsort(cent_mix)
            pressure5_1 = pressure5[ind_ans]
            pressure5_1 = pressure5_1[ind_ans_sort]

            p0 = [0,261.2571429,0]
            p1 = [609.6,278.6742857,0.3048]
            ind_ans = get_box(centroids,np.array([p0,p1]))
            cent_ind = centroids[ind_ans]
            cent_mix = cent_ind[:,0]
            ind_ans_sort = np.argsort(cent_mix)
            pressure5_2 = pressure5[ind_ans]
            pressure5_2 = pressure5_2[ind_ans_sort]

            pressure5 = 0.2*(pressure5_2-pressure5_1) + pressure5_1
            x5 = np.linspace(0,2000,35)
            tck = interpolate.splrep(x5,pressure5,s=0)
            p5 = interpolate.splev(x,tck,der=0)
            e5 = (sum((P-p5)**2)/(25*25))**(1/2)'''


        #    p_resp = np.linspace(0.623843,0,100)
        plt.figure(1)
        plt.title('t = 365 days')
        plt.plot(x1, pressure1_, 'r', x2, pressure2_, 'b', x3, pressure3_, 'g', x, P, 'y')
        plt.legend(('25 blocks', '225 blocks', '2025 blocks', 'Analytical Solution'))

        plt.grid()
        plt.ylabel('Pressure (psi)')
        plt.xlabel('Distance in X - Direction (ft)')
        plt.savefig('results/compositional/pressure_2d1_'  + '.png')

        plt.figure(2)
        plt.title('t = 365 days')
        plt.plot(x4, pressure4_, x, P)
        plt.ylabel('Pressure (psi)')
        plt.xlabel('Distance in X - Direction (ft)')
        plt.legend(('625 blocks', 'Analytical Solution'))
        plt.grid()
        plt.savefig('results/compositional/pressure_2d2_.png')

        plt.figure(3)
        plt.title('t = 365 days')
        plt.plot(x3, pressure3_, x, P)
        plt.ylabel('Pressure (psi)')
        plt.grid()
        plt.xlabel('Distance in X - Direction (ft)')
        plt.legend(('2025 blocks', 'Analytical Solution'))
        plt.savefig('results/compositional/pressure_2d3_.png')
        import pdb; pdb.set_trace()
