# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 14:06:18 2022

@author: RJ
"""
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp
from scipy.integrate import simps
import random
from joblib import Parallel, delayed
import multiprocessing
import pickle

num_cores = multiprocessing.cpu_count()

#data pulled from run data to compare against these calculations, slopes and resonance peaks
slope_bs = [1,1.25,1.5,2]
slopes = [4126394052.044609, 8698884758.364311, 15613382899.62825, 39144981412.639404]

peaks = np.array([0.4718,0.52957, .5492,.55326, 0.6284])*1e9+17.9e9
peak =  np.array([0.52957])*1e9+17.9e9
NePeaks = np.array([257142857.14285713,339285714.28571427, 472023809.52380955, 530357142.8571429,554166666.6666667])+17.9e9

#matplotlib needed a larger chunk size for some reason, may not be needed for others
mpl.rcParams['agg.path.chunksize'] = 100000

################ Constants #####################

mu = np.pi * 4 * (10**-7)
epsilon = 8.85418782 * (10**-12)
e = 1.602176634 * (10**-19)
c = 299792458
me = 9.10938356 * (10**-31) #kg
p11 = 1.841
a0 = 0.01156/2
kc = p11/a0
mc2 = 510998.950
JeV = 6.241509074e18
p01 = 2.405
pp01 = 3.832
k01 = p01/a0
delta = 0.172e-6
delta = 0.5e-6

################## Function defintions ######################
# A lot of these should be self explanatory, 
# but I will go into more detail for functions that are unique to my calculations

###### General CRES functions #######
def dfdt(power,energy, B):
    return -e*B*c**2*power/(2*np.pi)/(e*(mc2+energy))**2

def p_to_v(momentum):
    return momentum/np.sqrt(me**2+(momentum/c)**2)

def freq_to_energy(freq, field):
    return ((e*field/(me*2*np.pi*freq))-1)*mc2

def cyc_radius(energy, field, theta=90):
    gamma = energy / mc2 +1
    #print(gamma)
    p = np.sqrt(((energy+mc2)**2-mc2**2)/c**2) / JeV
    #print(p)
    v = p / (gamma * me)
    vp = v * np.sin(theta*np.pi / 180)
    return (gamma * me * vp)/(e*field)

def velocity(energy, field, theta=90):
    gamma = energy / mc2 +1
    #print(gamma)
    p = np.sqrt(((energy+mc2)**2-mc2**2)/c**2) / JeV
    #print(p)
    v = p / (gamma * me)
    vp = v * np.sin(theta*np.pi / 180)
    return vp

def cart_to_cyl(x,y,z):
        theta = np.arctan2(y, x)
        r = np.sqrt(x**2+y**2)
        return r, theta, z
    
def cyl_v(theta, vx, vy, vz):
    return (vx*np.cos(theta)+vy*np.sin(theta)), (vy*np.cos(theta)-vx*np.sin(theta)), vz
    

def rho_phi(center_x, center_y, time,omega, cyc_rad):
    xcoord = center_x + cyc_rad * np.cos(omega*time)
    ycoord = center_y + cyc_rad * np.sin(omega*time)
    rho = np.sqrt(xcoord**2 + ycoord**2)
    phi = np.arctan2(ycoord,xcoord)
    return (rho,phi)

######## Power Calculation functions ########

# creates perfectly circular trajectory for a given field, center position, and frequency
# outputs velocities, positions, times, and frequencies
# we output frequencies because the mode calculations can also take Kassiopeia trajectories as input, which have varying frequencies
def circle_traj_data(frequency, x0,y0,field):
    
    w = 2*np.pi*frequency
    energy = freq_to_energy(frequency, field)
    Rc = cyc_radius(energy, field)
    t = np.linspace(0,1/frequency,1000)
    x = Rc * np.cos(w*t) + x0
    y = Rc * np.sin(w*t) + y0
    z = 0*t
    
    vx = -Rc*w*np.sin(w*t)
    vy = Rc*w*np.cos(w*t)
    vz = 0*t
    
    f = frequency + 0*t
    
    r = np.sqrt(x**2+y**2)
    
    #check that event doesn't hit wall
    if np.any(r>a0):
        #print("track hit wall")
        return 0
    return [[vx,vy,vz], [x,y,z], t, f]

#general cutoff wavenumbers for TE and TM modes
def kcTM(n,m,a=a0):
    return sp.jn_zeros(n,m)[-1]/a

def kcTE(n,m,a=a0):
    return sp.jnp_zeros(n,m)[-1]/a

# general propogation constant for TE and TM modes (not relatavistic beta)
# if below cutoff returns 0 (this way the power will be reported as 0, instead of failing)
def TEbeta(w,n,m):
    kc = kcTE(n,m)
    k = w/c
    if k>kc:
        return np.sqrt(k**2-kc**2)
    else: 
        return 0
    
def TMbeta(w,n,m):
    kc = kcTM(n,m)
    k = w/c
    if k>kc:
        return np.sqrt(k**2-kc**2)
    else: 
        return 0
    
# propogation constants again but in a form I can utilize numpy linearization 
def TEbeta_fast(w,mode):
    kc = kcTE(mode[0],mode[1])
    k = w/c
    if k>kc:
        return np.sqrt(k**2-kc**2)
    else: 
        return 0
    
def TMbeta_fast(w,mode):
    kc = kcTM(mode[0],mode[1])
    k = w/c
    if k>kc:
        return np.sqrt(k**2-kc**2)
    else: 
        return 0
    
# lists all TE and TM modes by checking that betas are nonzero. 
# Checks up to n(m)=i(i+1) modes, this is justified in my report
#this version is slightly faster by using function mapping 
def mode_count(w, i=10):
    TE_modes = []
    TM_modes = []
    ns=np.arange(0,i)
    ms=np.arange(1,i+1)
    
    all_modes = np.array(np.meshgrid(ns,ms)).T.reshape(-1,2)
    
    TE_betas = np.array(list(map(lambda x: TEbeta_fast(w,x), all_modes)))
    TM_betas = np.array(list(map(lambda x: TMbeta_fast(w,x), all_modes)))
    
    TE_modes = all_modes[np.where(TE_betas)]
    TM_modes = all_modes[np.where(TM_betas)]
    

    return TE_modes, TM_modes

# this version only checks up to the estimated mode number from Dan's mode count paper
# see my report for some discussion of this. This is in fact faster than the 
# above version because I check fewer modes.
def mode_count3(w, i=10, verbose=False):
    TE_count = 0
    TE_modes = []
    TM_count = 0
    TM_modes = []
    ns=np.arange(0,i)
    ms=np.arange(1,i+1)
    Nc = mode_est(w/2/np.pi,1)
    print(Nc)
    for n in ns:
        print(TE_count+TM_count)
        if TE_count+TM_count >= Nc:
            return n-1
        for m in ms:
            if TEbeta(w,n,m)>0:
                TE_count+=1
                TE_modes.append((n,m))
            if TMbeta(w,n,m)>0:
                TM_count+=1
                TM_modes.append((n,m))

    return i

# oldest, slowest version, but nice when I want a list of all modes.
# I still use this when searching for resonance peaks for example. 
def mode_count2(w, i=10, verbose=False):
    TE_count = 0
    TE_modes = []
    TM_count = 0
    TM_modes = []
    ns=np.arange(0,i)
    ms=np.arange(1,i+1)
    
    for n in ns:
        for m in ms:
            if TEbeta(w,n,m)>0:
                TE_count+=1
                TE_modes.append((n,m))
            if TMbeta(w,n,m)>0:
                TM_count+=1
                TM_modes.append((n,m))
    if verbose:
        print("There are ", TE_count, " TE modes")
        print("There are ", TM_count, " TM modes")
    return TE_modes, TM_modes


def plot_mode_count(w,h):
    harmonics = np.arange(1,h)
    TMmodes = []
    TEmodes=[]
    for h in harmonics:
        TE, TM = mode_count(h*w, 30)
        TMmodes.append(len(TM))
        TEmodes.append(len(TE))
    plt.plot(harmonics, TMmodes, label="TM modes")
    plt.plot(harmonics, TEmodes, label="TE modes")
    plt.legend()
    plt.plot()
    return 

# cutoff calculations
def rect_cutoff(n,m):
    return c/(2*np.pi)*np.sqrt((m*np.pi/0.010668)**2+(n*np.pi/0.004318)**2)

def circ_cutoff(n,m):
    return kcTE(n,m)*c/2/np.pi

# cavity resonance calculations
def res_TE(n,m,l,a,d):
    t1 = kcTE(n,m,a)**2
    t2 = (l*np.pi/d)**2
    return c/2/np.pi*np.sqrt(t1+t2)

def res_TM(n,m,l,a,d):
    return c/2/np.pi*np.sqrt(kcTM(n,m,a)**2+(l*np.pi/d)**2)

# this functions estimates the number of modes we couple to from Dan's paper
def mode_est(f,h=1):
    f0 = 1.841*c/(2*np.pi*a0)
    return round(0.85 * (h*f/f0)**2)

# given a list of peak centroids attempt to find matches to harmonic cutoffs,
# must define how many harmonics you wish to check and tolerance
# playing with the tolerance is good since the exact radius is unknown until we've matched peaks
def resonance_search(n, tol, N=10):
    Peaks=peak
    TMPeaks = []
    TEPeaks = []
    TE = []
    TM = []
    TEh = []
    TMh = []
    matches = []
    matchPeaksM = []
    matchPeaksE = []
    ns=np.arange(0,n)
    ms=np.arange(1,n+1)
    for n in ns:
        for m in ms:
            fc0 = kcTE(n,m)*c/2/np.pi
            fc = fc0
            i = 1
            while fc > 19e9 and i<N:
                i+=1
                fc = fc0/i
            if fc > 18e9:
                TE.append((n,m))
                TEPeaks.append(fc)
                TEh.append(i)
                if any(np.abs(fc-Peaks) < tol):
                    matches.append([(n,m),fc,i, "E"])
                    matchPeaksE.append(fc)
                
            fc0 = kcTM(n,m)*c/2/np.pi
            fc = fc0
            i = 1
            while fc > 19e9 and i<N:
                i+=1
                fc = fc0/i
            if fc > 18e9:
                TM.append((n,m))
                TMPeaks.append(fc)
                TMh.append(i)
                if any(np.abs(fc-Peaks) < 1e7):
                    matches.append([(n,m),fc,i, "M"])
                    matchPeaksM.append(fc)

    plt.plot(TMPeaks, np.zeros(len(TMPeaks))+1, "x", label="TM cutoffs")
    plt.plot(TEPeaks, np.zeros(len(TEPeaks)),"x", label="TE cutoffs")
    plt.plot(matchPeaksM, np.zeros(len(matchPeaksM))+1, "x", label="TM macthes")
    plt.plot(matchPeaksE, np.zeros(len(matchPeaksE)),"x", label="TE matches")
    plt.plot(Peaks, np.zeros(len(Peaks))+2,"x", label="Observed Peaks")
    plt.legend()
    plt.xlabel("Frequency")
    plt.xlim(18e9,19e9)
    plt.show()
    return [TE, TM], [TEPeaks, TMPeaks], [TEh, TMh], matches


#Power from larmor formula
def PL(frequency, field, pitch = 90):
    energy = (frequency, field)
    gamma = energy / mc2 +1
    beta = np.sqrt(1-gamma**-2)
    rc = cyc_radius(energy, field)
    return (2*e**2*c*beta**4*gamma**4)/(12*np.pi*epsilon*rc**2)
    
#Total power that can propogate down guide for TM and TE modes (normalization constant)
def TE_P0(w, n, m):
    return np.pi*w*mu*TEbeta(w, n, m)/2/kcTE(n, m)**4*(sp.jnp_zeros(n,m)[-1]**2-n**2)*sp.jn(n, sp.jnp_zeros(n,m)[-1])**2

def TM_P0(w, n, m):
    return np.pi*w*epsilon*TMbeta(w, n, m)/2/kcTM(n, m)**4*(sp.jn_zeros(n,m)[-1])**2*(sp.jvp(n,sp.jn_zeros(n,m)[-1]))**2

#Rho and phi components of TE and TM fields
def TE_rho(rho, phi,w,A=0,B=1,n=1,m=1):
    return -1j*w*mu*n/(kcTE(n,m)**2*rho)*(A*np.cos(n*phi)-B*np.sin(n*phi))*sp.jv(n,kcTE(n,m)*rho)

def TE_phi(rho, phi,w,A=0,B=1,n=1, m=1):
    return 1j*w*mu/kcTE(n,m)*(A*np.sin(n*phi)+B*np.cos(n*phi))*sp.jvp(n,kcTE(n,m)*rho)

def TM_rho(rho, phi,w,A=1,B=0,n=0,m=1):
    return -1j*TMbeta(w,n,m)/kcTM(n,m)*(A*np.sin(n*phi)+B*np.cos(n*phi))*sp.jvp(n,kcTM(n,m)*rho)

def TM_phi(rho, phi,w,A=1,B=0,n=0,m=1):
    return -1j*TMbeta(w,n,m)*n/(kcTM(n,m)**2*rho)*(A*np.cos(n*phi)-B*np.sin(n*phi))*sp.jv(n,kcTM(n,m)*rho)

#power coupling for given trajectory for TE and TM modes
def TMnm(vs, positions,times,freqs, n=0, m=1, h=1):
    ws = h*2*np.pi*freqs
    # need an average w for fourier transform 
    w = np.average(ws)
    
    rhos, phis, zs = cart_to_cyl(*positions)
    
    Js = [-e*v for v in vs]
    
    # calculate fields with cylindrical coordinates
    Ea_rho = TM_rho(rhos, phis, w,1,0,n,m)
    Ea_phi = TM_phi(rhos, phis, w,1,0,n,m)
    Eb_rho = TM_rho(rhos, phis, w,0,1,n,m)
    Eb_phi = TM_phi(rhos, phis, w,0,1,n,m)
    
    #convert to cartesian and add expponential term
    Exa = (Ea_rho*np.cos(phis) - Ea_phi*np.sin(phis))*np.exp(-1j*w*times)
    Eya = (Ea_rho*np.sin(phis) + Ea_phi*np.cos(phis))*np.exp(-1j*w*times)
    
    Exb = (Eb_rho*np.cos(phis) - Eb_phi*np.sin(phis))*np.exp(-1j*w*times)
    Eyb = (Eb_rho*np.sin(phis) + Eb_phi*np.cos(phis))*np.exp(-1j*w*times)
    
    #normalization constant
    PN = TM_P0(w,n,m)
    
    #amplitude from each polarization, since rotationally symmetric I just choose x and y axis to be polarizations
    Ab = -simps((Exb*Js[0]+Eyb*Js[1]), times)/(PN*times[-1])
    Aa = -simps((Exa*Js[0]+Eya*Js[1]), times)/(PN*times[-1])
    #print(Aa,Ab)
    #return power by adding each polarizations contribution
    return (np.absolute(Ab)**2+np.absolute(Aa)**2)*PN


def TEnm(vs, positions,times,freqs, n=1, m=1, h=1):
    ws = h*2*np.pi*freqs
    w = np.average(ws)
    if TEbeta(w,n,m) == 0:
        return 0
    rhos, phis, zs = cart_to_cyl(*positions)
    
    Js = [-e*v for v in vs]
    
    Ea_rho = TE_rho(rhos, phis, w,1,0,n,m)
    Ea_phi = TE_phi(rhos, phis, w,1,0,n,m)
    
    Eb_rho = TE_rho(rhos, phis, w,0,1,n,m)
    Eb_phi = TE_phi(rhos, phis, w,0,1,n,m)
    
    Exb = (Eb_rho*np.cos(phis) - Eb_phi*np.sin(phis))*np.exp(-1j*w*times)
    Eyb = (Eb_rho*np.sin(phis) + Eb_phi*np.cos(phis))*np.exp(-1j*w*times)
    
    Exa = (Ea_rho*np.cos(phis) - Ea_phi*np.sin(phis))*np.exp(-1j*w*times)
    Eya = (Ea_rho*np.sin(phis) + Ea_phi*np.cos(phis))*np.exp(-1j*w*times)
    
    PN = TE_P0(w,n,m)
    
    Ab = -simps((Exb*Js[0]+Eyb*Js[1]), times)/(PN*times[-1])
    Aa = -simps((Exa*Js[0]+Eya*Js[1]), times)/(PN*times[-1])
    return (np.absolute(Ab)**2+np.absolute(Aa)**2)*PN


##### OLD Analysis Functions ######
# the following functions should work but are not optimized to be fast
# they may not be parallelized, check the ideal number of modes (most must be manually set), or be user friendly
# I'm leaving them in here in case we need them in the future, at which point I can update them


#plot TEnm power at different frequencies and fields
def TEnm_rho(frequencies,rhos0,field,n=1,m=1,h=1):
    for f in frequencies:
        p = []
        rhos = max_rho(f, rhos0, field,len(rhos0))
        for r in rhos:
            p.append(TEnm(*circle_traj_data(f, r, 0, field),n,m,h))
        plt.plot(rhos, p, label=str(f))
    plt.legend()
    plt.xlabel("rho_c")
    plt.ylabel("Power")
    plt.title("Power in TE"+ str(n)+ str(m)+"("+str(h)+")"+" mode at "+ str(field)+ "T")
    plt.show()
    return

#plot TMnm power at different frequencies and fields
def TMnm_rho(frequencies,rhos0,field,n=1,m=1,h=1):
    for f in frequencies:
        p = []
        rhos = max_rho(f, rhos0, field,len(rhos0))
        for r in rhos:
            #print(r)
            #print(circle_traj_data(f, r, 0, field)==0)
            p.append(TMnm(*circle_traj_data(f, r, 0, field),n,m,h))
        plt.plot(rhos, p, label=str(f))
    plt.legend()
    plt.xlabel("rho_c")
    plt.ylabel("Power")
    plt.title("Power in TM"+ str(n)+ str(m)+"("+str(h)+")"+" mode at "+ str(field)+ "T")
    plt.show()
    return

#plot total power of all TE modes at given harmonic
def TEh_rho(frequencies,rhos,field,h=1):
    for f in frequencies:
        TEmodes, TMmodes = mode_count(f*h*2*np.pi, 20)
        ps = []
        for r in rhos:
            P = 0
            for mode in TEmodes:
                Pnm = (TEnm(*circle_traj_data(f, r, 0, field),mode[0],mode[1],h))
                if Pnm < 0: print(Pnm)
                P += Pnm
            ps.append(P)
        plt.plot(rhos, ps, label=str(f))
    plt.legend()
    plt.xlabel("rho_c")
    plt.ylabel("Power")
    #plt.title("Power in T"+ str(n)+ str(m)+" mode at"+ str(field)+ "T")
    plt.show()
    return

#plot total power of all TE modes up to a given harmonic
def TEt_rho(frequencies,rhos0,field,h=1):
    for f in frequencies:
        ps = []
        rhos = max_rho(f, rhos0, field,len(rhos0))
        for r in rhos:
            P = 0
            for harmonic in np.arange(1,h+1):
                #print(harmonic)
                TEmodes, TMmodes = mode_count(f*harmonic*2*np.pi, 20)
                for mode in TEmodes:
                    P += (TEnm(*circle_traj_data(f, r, 0, field),mode[0],mode[1],harmonic))
            ps.append(P)
        plt.plot(rhos, ps, label=str(f))
    #plt.hlines([PL(f, field) for f in frequencies], 0, 0.0035)
    plt.legend()
    plt.xlabel("rho_c")
    plt.ylabel("Power")
    plt.title("Power in TE modes up to " + str(h)+ " harmonic at" + str(field)+ "T")
    plt.show()
    return

#plot total power of all modes up to a given harmonic
def Pt_rho(frequencies,rhos0,field,h=1,N=20):
    for f in frequencies:
        ps = []
        rhos = max_rho(f, rhos0, field,len(rhos0))
        for r in rhos:
            P = 0
            for harmonic in np.arange(1,h+1):
                #print(harmonic)
                TEmodes, TMmodes = mode_count(f*harmonic*2*np.pi, N)
                for mode in TEmodes:
                    P += (TEnm(*circle_traj_data(f, r, 0, field),mode[0],mode[1],harmonic))
                for mode in TMmodes:
                    P += (TMnm(*circle_traj_data(f, r, 0, field),mode[0],mode[1],h))
            ps.append(P)
        plt.plot(rhos, ps, label=str(f))
    #plt.hlines([PL(f, field) for f in frequencies], 0, 0.0035)
    plt.legend()
    plt.xlabel("rho_c")
    plt.ylabel("Power")
    plt.title("Power in all modes up to " + str(h)+ " harmonic at" + str(field)+ "T")
    plt.show()
    return

#plot total power of all TM modes at given harmonic
def TMh_rho(frequencies,rhos,field,h=1):
    for f in frequencies:
        TEmodes, TMmodes = mode_count(f*h*2*np.pi, 20)
        ps = []
        for r in rhos:
            P = 0
            for mode in TMmodes:
                Pnm = (TMnm(*circle_traj_data(f, r, 0, field),mode[0],mode[1],h))
                if Pnm < 0: print(Pnm)
                P += Pnm
            ps.append(P)
        plt.plot(rhos, ps, label=str(f))
    plt.legend()
    plt.xlabel("rho_c")
    plt.ylabel("Power")
    #plt.title("Power in T"+ str(n)+ str(m)+" mode at"+ str(field)+ "T")
    plt.show()
    return

#plot total power of all TM modes up to a given harmonic
def TMt_rho(frequencies,rhos0,field,h=1):
    for f in frequencies:
        ps = []
        rhos = max_rho(f, rhos0, field,len(rhos0))
        for r in rhos:
            P = 0
            for harmonic in np.arange(1,h+1):
                #print(harmonic)
                TEmodes, TMmodes = mode_count(f*harmonic*2*np.pi, 20)
                for mode in TMmodes:
                    P += (TMnm(*circle_traj_data(f, r, 0, field),mode[0],mode[1],harmonic))
            ps.append(P)
        plt.plot(rhos, ps, label=str(f))
    plt.legend()
    plt.xlabel("rho_c")
    plt.ylabel("Power")
    #plt.title("Power in T"+ str(n)+ str(m)+" mode at"+ str(field)+ "T")
    plt.show()
    return

#plot total power of all TE modes up to a given harmonic
def TEt_vs_h(frequencies,rhos0,field, h=1):
    for f in frequencies:
        ps_ave = []
        rho_max = 0
        for rho in rhos0:
            if circle_traj_data(f, rho, 0, field) != 0:
                rho_max = rho
        rhos = np.linspace(0,rho_max, 10)
        print(rho_max)
        Nrho = np.sum(np.array(rhos)**2)
        for harmonic in np.arange(1,h+1):
            P = 0
            for r in rhos:
                #P = 0
                #print(harmonic)
                TEmodes, TMmodes = mode_count(f*harmonic*2*np.pi, 20)
                for mode in TEmodes:
                    P +=  rho**2 * (TEnm(*circle_traj_data(f, r, 0, field),mode[0],mode[1],harmonic)) / Nrho
                #ps.append(P)
            ps_ave.append(P)
        plt.plot(np.arange(1,h+1), ps_ave, label=str(f))
    plt.legend()
    plt.xlabel("harmonic")
    plt.ylabel("Power")
    plt.title("Power in TE modes at " + str(field)+ "T vs harmonic")
    plt.show()
    return


# plot total TM power vs harmonic
def TMt_vs_h(frequencies,rhos0,field, h=1):
    for f in frequencies:
        ps_ave = []
        rho_max = 0
        for rho in rhos0:
            if circle_traj_data(f, rho, 0, field) != 0:
                rho_max = rho
        rhos = np.linspace(0,rho_max, 10)
        print(rho_max)
        Nrho = np.sum(np.array(rhos)**2)
        for harmonic in np.arange(1,h+1):
            P = 0
            for r in rhos:
                #P = 0
                #print(harmonic)
                TEmodes, TMmodes = mode_count(f*harmonic*2*np.pi, 20)
                for mode in TMmodes:
                    P +=  rho**2 * (TMnm(*circle_traj_data(f, r, 0, field),mode[0],mode[1],harmonic)) / Nrho
                #ps.append(P)
            ps_ave.append(P)
        plt.plot(np.arange(1,h+1), ps_ave, label=str(f))
    plt.legend()
    plt.xlabel("harmonic")
    plt.ylabel("Power")
    plt.title("Power in TM modes at " + str(field)+ "T vs harmonic")
    plt.show()
    return
        
#plot total TE power vs field 
def TEt_vs_B(frequencies, rhos0, fields, h=1, larmor = 0, slope = 0, data=0):
    for f in frequencies:
        Ps = []
        for field in fields:
            print(field)
            rho_max = 0
            for rho in rhos0:
                if circle_traj_data(f, rho, 0, field) != 0:
                    rho_max = rho
            rhos = np.linspace(0,rho_max, 50)
            print(rho_max)
            Nrho = np.sum(np.array(rhos))
            P=0
            for harmonic in np.arange(1,h+1):
                TEmodes, TMmodes = mode_count(f*harmonic*2*np.pi, 20)
                for mode in TEmodes:
                    for rho in rhos:
                        P += rho * (TEnm(*circle_traj_data(f, rho, 0, field),mode[0],mode[1],harmonic)) / Nrho
            Ps.append(P)
        if slope:
            plt.plot(fields, dfdt(-1*np.array(Ps), freq_to_energy(f,fields),fields), label=str(f))
            if data:
                plt.plot(slope_bs, slopes,".", label="Ne data")
            if larmor:
                plt.plot(fields, dfdt(-1*PL(f,fields), freq_to_energy(f,fields),fields), label="Larmor " + str(f))
        else:
            plt.plot(fields, Ps, label=str(f))
            if larmor:
                plt.plot(fields, PL(f, fields), label="Larmor " + str(f))
    plt.legend()
    plt.xlabel("B")
    if slope:
        plt.ylabel("Slope")
        plt.title("Slope from TE modes up to " + str(h)+ " harmonic at vs field")
    else:
        plt.ylabel("Power")
        plt.title("Power in TE modes up to " + str(h)+ " harmonic at vs field")
    
    plt.show()
    return

###### Current Analysis Functions #######
# I've also left their older counterparts in this section so its easier to compare what changed

#calculate total power up to given harmonic
def Pt(field, rho, f,h,N):
    P=0
    
    for harmonic in np.arange(1,h+1):
        #see report for why I search up to 2h modes
        TEmodes, TMmodes = mode_count(f*harmonic*2*np.pi, harmonic*2)
        trajectory = circle_traj_data(f, rho, 0, field)
        # Calculates each mode in parallel to save time, returns list of powers
        pm = Parallel(n_jobs=num_cores)(delayed(TMnm)(*trajectory,mode[0],mode[1],harmonic) for mode in TMmodes)
        pe = Parallel(n_jobs=num_cores)(delayed(TEnm)(*trajectory,mode[0],mode[1],harmonic) for mode in TEmodes)
        #add up mode powers and add to total
        P += np.sum(pm) + np.sum(pe)
    return P

#old slower version
def Pt2(field, rho, f,h,N):
    P=0
    for harmonic in np.arange(1,h+1):
        TEmodes, TMmodes = mode_count(f*harmonic*2*np.pi, N)
        p=0
        for mode in TMmodes:
            p += (TMnm(*circle_traj_data(f, rho, 0, field),mode[0],mode[1],harmonic)) 
        for mode in TEmodes:
            p += (TEnm(*circle_traj_data(f, rho, 0, field),mode[0],mode[1],harmonic)) 
        P += p
    return P

# Total power vs harmonic (parallelizing didn't seem to work in this case)
def Pt_vs_h(frequencies,rhos0,field, h=1):
    for f in frequencies:
        ps_ave = []
        rhos = max_rho(f, rhos0, field,len(rhos0))
        Nrho = np.sum(np.array(rhos))
        
        for harmonic in np.arange(1,h+1):
            print(harmonic)
            P = 0
            for rho in rhos:
                TEmodes, TMmodes = mode_count(f*harmonic*2*np.pi, harmonic*2)
                for mode in TMmodes:
                    P += rho*(TMnm(*circle_traj_data(f, rho, 0, field),mode[0],mode[1],harmonic)) / Nrho
                for mode in TEmodes:
                    P += rho*(TEnm(*circle_traj_data(f, rho, 0, field),mode[0],mode[1],harmonic)) / Nrho
                # trajectory = circle_traj_data(f, rho, 0, field)
                # ptm = np.sum(Parallel(n_jobs=10)(delayed(TMnm)(*trajectory,mode[0],mode[1],harmonic) for mode in TMmodes))
                # pte = np.sum(Parallel(n_jobs=10)(delayed(TEnm)(*trajectory,mode[0],mode[1],harmonic) for mode in TEmodes))
                # #print(ptm,pte)
                # P += rho*ptm/ Nrho
                # P += rho*pte/ Nrho
                
            ps_ave.append(P)
        plt.plot(np.arange(1,h+1), ps_ave, label=str(f))
    plt.legend()
    plt.xlabel("harmonic")
    plt.ylabel("Power")
    plt.title("Power at " + str(field)+ "T vs harmonic")
    plt.show()
    
    save_data = {'hs':np.arange(1,h+1),'powers':ps_ave, 'field':field}
    filename = "PvsH" + str(field)+ "T"+ ".pkl"
    with open(filename,'wb') as file:
        pickle.dump(save_data,file)

    return

#total power vs field, with options to add larmor power, convert to slopes, and add real slope data
def Pt_vs_B(frequencies, rhos0, fields, h=1, larmor = 0, slope = 0, data=0):
    Powers = []
    for f in frequencies:
        Ps = []
        for field in fields:
            print(field)
            rhos = max_rho(f, rhos0, field,len(rhos0))
            Nrho = np.sum(np.array(rhos))
            P=0
            
            for rho in rhos:
                P += rho*Pt(field, rho, f,h,2*h)/Nrho
                Ps.append(P)
        if slope:
            plt.plot(fields, dfdt(-1*np.array(Ps), freq_to_energy(f,fields),fields), label=str(f))
            if data:
                plt.plot(slope_bs, slopes,".", label="Ne data")
            if larmor:
                plt.plot(fields, dfdt(-1*PL(f,fields), freq_to_energy(f,fields),fields), label="Larmor " + str(f))
        else:
            plt.plot(fields, Ps, label=str(f))
            if larmor:
                plt.plot(fields, PL(f, fields), label="Larmor " + str(f))
        Powers.append(Ps)
    plt.legend()
    plt.xlabel("B")
    if slope:
        plt.ylabel("Slope")
        plt.title("Slope from All modes up to " + str(h)+ " harmonic at vs field")
    else:
        plt.ylabel("Power")
        plt.title("Power in All modes up to " + str(h)+ " harmonic at vs field")
    
    plt.show()
    return Powers

#old slow version
def Pt_vs_B2(frequencies, rhos0, fields, h=1, larmor = 0, slope = 0, data=0, N=20):
    Powers = []
    for f in frequencies:
        Ps = []
        for field in fields:
            print(field)
            rhos = max_rho(f, rhos0, field,len(rhos0))
            Nrho = np.sum(np.array(rhos))
            P=0
            
            for harmonic in np.arange(1,h+1):
                TEmodes, TMmodes = mode_count(f*harmonic*2*np.pi, N)
                for mode in TMmodes:
                    for rho in rhos:
                        P += rho * (TMnm(*circle_traj_data(f, rho, 0, field),mode[0],mode[1],harmonic)) / Nrho
                for mode in TEmodes:
                    for rho in rhos:
                        P += rho * (TEnm(*circle_traj_data(f, rho, 0, field),mode[0],mode[1],harmonic)) / Nrho
            Ps.append(P)
        if slope:
            plt.plot(fields, dfdt(-1*np.array(Ps), freq_to_energy(f,fields),fields), label=str(f))
            if data:
                plt.plot(slope_bs, slopes,".", label="Ne data")
            if larmor:
                plt.plot(fields, dfdt(-1*PL(f,fields), freq_to_energy(f,fields),fields), label="Larmor " + str(f))
        else:
            plt.plot(fields, Ps, label=str(f))
            if larmor:
                plt.plot(fields, PL(f, fields), label="Larmor " + str(f))
        Powers.append(Ps)
    plt.legend()
    plt.xlabel("B")
    if slope:
        plt.ylabel("Slope")
        plt.title("Slope from All modes up to " + str(h)+ " harmonic at vs field")
    else:
        plt.ylabel("Power")
        plt.title("Power in All modes up to " + str(h)+ " harmonic at vs field")
    
    plt.show()
    return Powers
        
#plot power of single TE mode up to a given harmonic vs rho
def TEnm_rho_h(frequencies,rhos,field,n=1,m=1,h=1):
    for f in frequencies:
        ps = []
        ps1 =[]
        for r in rhos:
            P = 0
            for harmonic in np.arange(1,h+1):
                P += (TEnm(*circle_traj_data(f, r, 0, field),n,m,harmonic))
            ps.append(P)
            ps1.append(TEnm(*circle_traj_data(f, r, 0, field),n,m,1))
        plt.plot(rhos, ps, label=str(f))
        plt.plot(rhos, ps1, label=str(f)+" fundamental")
    #plt.hlines([PL(f, field) for f in frequencies], 0, 0.0035)
    plt.legend()
    plt.xlabel("rho_c")
    plt.ylabel("Power")
    plt.title("Power in TE11 mode up to " + str(h)+ " harmonic at " + str(field)+ " T")
    plt.show()
    return

#make quiver and line plots for any given mode
def mode_plot(n,m,f, TM = 0):
        xs = np.linspace(-2*a0,2*a0,40)
        ys = np.linspace(-2*a0, 2*a0, 40)
        
        X, Y = np.meshgrid(xs, ys)
        
        Rho = np.sqrt(X**2 + Y**2)
        Phi = np.arctan2(Y,X)
        
        theta = np.linspace(0, 2*np.pi, 100)
        a = a0*np.cos(theta)
        b = a0*np.sin(theta)
        
        w = 2*np.pi*f
        
        if TM:
            Ea_rho = TM_rho(Rho, Phi, w,0,1,n,m)
            Ea_phi = TM_phi(Rho, Phi, w,0,1,n,m)
            
        else:
            Ea_rho = TE_rho(Rho, Phi, w,0,1,n,m)
            Ea_phi = TE_phi(Rho, Phi, w,0,1,n,m)
            
        Exa = (1j*Ea_rho*np.cos(Phi) - 1j*Ea_phi*np.sin(Phi))
        Eya = (1j*Ea_rho*np.sin(Phi) + 1j*Ea_phi*np.cos(Phi))
        
        
        plt.quiver(X,Y, Exa, Eya)
        plt.xlim(-a0,a0)
        plt.ylim(-a0,a0)
        plt.plot(a,b,"red")
        axes=plt.gca()
        axes.set_aspect(1)
        plt.title("TE mode "+str(n)+","+str(m))
        plt.show()
        
        xs = np.linspace(-2*a0,2*a0,100)
        ys = np.linspace(-2*a0, 2*a0, 100)
        
        X, Y = np.meshgrid(xs, ys)
        
        Rho = np.sqrt(X**2 + Y**2)
        Phi = np.arctan2(Y,X)
        
        w = 2*np.pi*f
        
        if TM:
            Ea_rho = TM_rho(Rho, Phi, w,0,1,n,m)
            Ea_phi = TM_phi(Rho, Phi, w,0,1,n,m)
            
        else:
            Ea_rho = TE_rho(Rho, Phi, w,0,1,n,m)
            Ea_phi = TE_phi(Rho, Phi, w,0,1,n,m)
            
        Exa = (1j*Ea_rho*np.cos(Phi) - 1j*Ea_phi*np.sin(Phi))
        Eya = (1j*Ea_rho*np.sin(Phi) + 1j*Ea_phi*np.cos(Phi))
        
        plt.streamplot(X,Y,np.real(Exa),np.real(Eya), density=2)
        plt.xlim(-a0,a0)
        plt.ylim(-a0,a0)
        plt.plot(a,b,"red")
        axes=plt.gca()
        axes.set_aspect(1)
        plt.title("TE mode "+str(n)+","+str(m))
        plt.show()
        
        return
            
        
# TE power of specific mode vs frequency (nice for showing cutoff effect)        
def TEnm_vs_f(frequencies, rhos0, field,n=1,m=1, h=1):
    rho_max = 0
    for rho in rhos0:
        if circle_traj_data(18e9, rho, 0, field) != 0:
            rho_max = rho
    rhos = np.linspace(0,rho_max, 50)
    print(rho_max)
    Nrho = np.sum(np.array(rhos))
    Ps = []
    for f in frequencies:
        P=0
        for rho in rhos:
            P += rho * (TEnm(*circle_traj_data(f, rho, 0, field),n, m,h)) / Nrho
        Ps.append(P)
        
    plt.plot(frequencies, Ps)
    plt.xlabel("frequency")
    plt.ylabel("Power")
    plt.title("Power from TE"+ str(n) + str(m) + " mode at " + str(h)+ " harmonic vs frequency") 
    plt.show()
    return

def TMnm_vs_f(frequencies, rhos0, field,n=1,m=1, h=1):
    rho_max = 0
    for rho in rhos0:
        if circle_traj_data(18e9, rho, 0, field) != 0:
            rho_max = rho
    rhos = np.linspace(0,rho_max, 50)
    print(rho_max)
    Nrho = np.sum(np.array(rhos)**2)
    Ps = []
    for f in frequencies:
        P=0
        for rho in rhos:
            P += rho**2 * (TMnm(*circle_traj_data(f, rho, 0, field),n, m,h)) / Nrho
        Ps.append(P)
        
    plt.plot(frequencies, Ps)
    plt.xlabel("frequency")
    plt.ylabel("Power")
    plt.title("Power from TE"+ str(n) + str(m) + " modes at " + str(h)+ " harmonic vs frequency") 
    plt.show()
    return

#returns modified list of rhos removing values that would hit wall
def max_rho(f, rhos0, field,length):
    rho_max = 0
    for rho in rhos0:
        if circle_traj_data(f, rho, 0, field) != 0:
            rho_max = rho
    return np.linspace(0,rho_max,length)

#plot total power in each mode
def Pt_mode_hist(frequency, rhos0, fields, h=1, N=20):
    plt.figure(figsize=(30,10))
    powers = []
    for field in fields:
        print(field)
        TMPs = []
        Mmodes = []
        TEPs = []
        Emodes = []
        rhos = max_rho(frequency, rhos0, field, len(rhos0))
        Nrho = np.sum(np.array(rhos)**2)
        for harmonic in np.arange(1,h+1):
            TEmodes, TMmodes = mode_count(frequency*harmonic*2*np.pi, N)
            for mode in TMmodes:
                P= 0
                for rho in rhos:
                    P += rho**2 * (TMnm(*circle_traj_data(frequency, rho, 0, field),mode[0],mode[1],harmonic)) / Nrho
                if mode not in Mmodes:
                    TMPs.append(P)
                    Mmodes.append(mode)
                else:
                    TMPs[Mmodes.index(mode)]+=P
                
            for mode in TEmodes:
                P=0
                for rho in rhos:
                    P += rho**2 * (TEnm(*circle_traj_data(frequency, rho, 0, field),mode[0],mode[1],harmonic)) / Nrho
                if mode not in Emodes:
                    TEPs.append(P)
                    Emodes.append(mode)
                else:
                    TEPs[Emodes.index(mode)]+=P
        
        # plt.plot(np.arange(0,len(TMPs+TEPs)), TMPs+TEPs,"o", label=str(field) + "T")
        plt.bar(np.arange(0,len(TMPs+TEPs)), TMPs+TEPs, label=str(field) + "T", alpha=0.6)
        powers.append(TMPs+TEPs)
    
    xticks = ["TM"+str(mode[0])+ str(mode[1]) for mode in Mmodes] + ["TE"+str(mode[0])+ str(mode[1]) for mode in Emodes]

    plt.xticks(np.arange(0,len(TEPs+TMPs)),xticks)
    plt.legend()
    plt.xlabel("Mode")
    plt.ylabel("Power")
    plt.title("Power from All modes up to " + str(h)+ " harmonic")
    plt.show()
    return powers, xticks
        
        
#generate expected wedge plot at given field, currently only works for 
#90 degree pitch angle and doesn't handle frequency distribution correctly
def wedge_plot(field, events, fs = [18e9,19e9], N=20,h=20, state=0, pitch=90):
    if state:
        random.setstate(state)
    else:
        random.seed()
        state = random.getstate()
    
    freqs = []
    p11 = []
    slopes = []
    for event in range(events):
        f = random.uniform(fs[0],fs[1])
        r = random.uniform(0,max_rho(f, np.linspace(0,a0,100), field, 100)[-1])
        print("frequency",f)
        print("Rho",r)
        freqs.append(f)
        p11.append(TEnm(*circle_traj_data(f, r, 0, field),1,1,1))
        P =  Pt(field, r, f,h,N)
        slopes.append(dfdt(-P,freq_to_energy(f, field),field))
    
    plot = plt.scatter(slopes,p11,c=freqs,cmap="viridis")
    cb = plt.colorbar(plot)
    plt.ylim(0,1e-14)
    plt.show()
    return state

#histogram of expected slopes for uniform freq spectrum
def slope_hist_mono(field,events, fs = [18e9,19e9], N=12,h=20, state=0):
   if state:
       random.setstate(state)
   else:
       random.seed()
       state = random.getstate()
       
   freqs = [random.uniform(fs[0],fs[1]) for _ in range(events)]
   rhos=[]
   slopes = []
   for f in freqs:
       rm = max_rho(f, np.linspace(0,a0,100), field, 100)[-1]
       rho = random.triangular(0,rm,rm)
       rhos.append(rho)
       P = Pt(field, rho, f,h,N)
       slopes.append(dfdt(-P,freq_to_energy(f, field),field))
   
   save_data = {'freqs':freqs,'slopes':slopes,'rhos':rhos,'field':field, 'state':state, 'N':N, 'h':h}
   filename = "SlopeHist" + str(field)+ "T"+ str(events)+ "E" + ".pkl"
   with open(filename,'wb') as file:
       pickle.dump(save_data,file)
   
   plt.hist(slopes,bins=100)
   plt.xlabel("slope")
   plt.ylabel("counts")
   plt.title("Slope histogram at "+ str(field)+ "T for "+ str(events)+ " events")
   #plt.show()
   return save_data       

# old slower version
def slow_slope_hist_mono(field,events, fs = [18e9,19e9], N=12,h=20, state=0):
   if state:
       random.setstate(state)
   else:
       random.seed()
       state = random.getstate()
       
   freqs = [random.uniform(fs[0],fs[1]) for _ in range(events)]
   rhos=[]
   slopes = []
   for f in freqs:
       rm = max_rho(f, np.linspace(0,a0,100), field, 100)[-1]
       rho = random.triangular(0,rm,rm)
       rhos.append(rho)
       P = Pt2(field, rho, f,h,N)
       slopes.append(dfdt(-P,freq_to_energy(f, field),field))
   
   save_data = {'freqs':freqs,'slopes':slopes,'rhos':rhos,'field':field, 'state':state, 'N':N, 'h':h}
   filename = "SlopeHist" + str(field)+ "T"+ str(events)+ "E" + ".pkl"
   with open(filename,'wb') as file:
     pickle.dump(save_data,file)

   plt.hist(slopes,bins=100)
   plt.xlabel("slope")
   plt.ylabel("counts")
   plt.title("Slope histogram at "+ str(field)+ "T for "+ str(events)+ " events")
   #plt.show()
   return save_data   

# test to investigate fft dependence on radius
def JEfft_test(N,M,Rho):
    f=18e9
    field=3
    field2=1
    x0=Rho
    y0=0
    n=N
    m=M
    
    w = 2*np.pi*f
    energy = freq_to_energy(f, field)
    Rc = cyc_radius(energy, field)
    
    energy2 = freq_to_energy(f, field2)
    Rc2 = cyc_radius(energy2, field)
    gamma = energy / mc2 +1
    p = np.sqrt(((energy+mc2)**2-mc2**2)/c**2) / JeV
    field3 = p/(e*Rc2)
    f2 = e*field3/(me*2*np.pi*gamma)
    w2 = 2*np.pi*f2
    print(f/1e9,f2/1e9)
    
    f2=f
    w2=w
    
    t1 = np.linspace(0,1/f,1000)
    t2 = np.linspace(0,1/f2,1000)
    
    x1 = Rc * np.cos(w*t1) + x0
    y1 = Rc * np.sin(w*t1) + y0
    z1 = 0*t1
    
    x2 = Rc2 * np.cos(w2*t2) + x0
    y2 = Rc2 * np.sin(w2*t2) + y0
    z2 = 0*t2
    
    vx1 = -Rc*w*np.sin(w*t1)
    vy1 = Rc*w*np.cos(w*t1)
    vz1 = 0*t1
    
    vx2 = -Rc2*w2*np.sin(w2*t2)
    vy2 = Rc2*w2*np.cos(w2*t2)
    vz2 = 0*t2
    
    positions1 = [x1,y1,z1]
    vs1 = [vx1,vy1,vz1]
    positions2 = [x2,y2,z2]
    vs2 = [vx2,vy2,vz2]
    
    rhos1, phis1, zs = cart_to_cyl(*positions1)
    rhos2, phis2, zs = cart_to_cyl(*positions2)
    
    Js1 = [-e*v for v in vs1]
    Js2 = [-e*v for v in vs2]
    
    fft1a = []
    fft1b =[]
    fft2a=[]
    fft2b=[]
    
    for h in range(100):
        Ea_rho1 = TE_rho(rhos1, phis1, h*w,1,0,n,m)
        Ea_phi1 = TE_phi(rhos1, phis1, h*w,1,0,n,m)
        Eb_rho1 = TE_rho(rhos1, phis1, h*w,0,1,n,m)
        Eb_phi1 = TE_phi(rhos1, phis1, h*w,0,1,n,m)
        
        Exa1 = (Ea_rho1*np.cos(phis1) - Ea_phi1*np.sin(phis1))*np.exp(-1j*w*h*t1)
        Eya1 = (Ea_rho1*np.sin(phis1) + Ea_phi1*np.cos(phis1))*np.exp(-1j*w*h*t1)
        
        Exb1 = (Eb_rho1*np.cos(phis1) - Eb_phi1*np.sin(phis1))*np.exp(-1j*w*h*t1)
        Eyb1 = (Eb_rho1*np.sin(phis1) + Eb_phi1*np.cos(phis1))*np.exp(-1j*w*h*t1)
        
        Ea_rho2 = TE_rho(rhos2, phis2, h*w2,1,0,n,m)
        Ea_phi2 = TE_phi(rhos2, phis2, h*w2,1,0,n,m)
        Eb_rho2 = TE_rho(rhos2, phis2, h*w2,0,1,n,m)
        Eb_phi2 = TE_phi(rhos2, phis2, h*w2,0,1,n,m)
        
        Exa2 = (Ea_rho2*np.cos(phis2) - Ea_phi2*np.sin(phis2))*np.exp(-1j*w*h*t2)
        Eya2 = (Ea_rho2*np.sin(phis2) + Ea_phi2*np.cos(phis2))*np.exp(-1j*w*h*t2)
        
        Exb2 = (Eb_rho2*np.cos(phis2) - Eb_phi2*np.sin(phis2))*np.exp(-1j*w*h*t2)
        Eyb2 = (Eb_rho2*np.sin(phis2) + Eb_phi2*np.cos(phis2))*np.exp(-1j*w*h*t2)
        
        JEB1 = Exb1*Js1[0]+Eyb1*Js1[1]
        JEB2 = Exb2*Js2[0]+Eyb2*Js2[1]
        JEA1 = Exa1*Js1[0]+Eya1*Js1[1]
        JEA2 = Exa2*Js2[0]+Eya2*Js2[1]
        
        JEA2 = Exa2*Js1[0]+Eya2*Js1[1]
        JEB2 = Exb2*Js1[0]+Eyb2*Js1[1]
    
        fft1b.append(-simps(JEB1, t1))
        fft1a.append(-simps(JEA1, t1))
        fft2b.append(-simps(JEB2, t2))
        fft2a.append(-simps(JEA2, t2))
    
    plt.plot(np.arange(len(fft1a)),np.absolute(fft1a), label="Rc("+str(field)+")",alpha=.5)
    plt.plot(np.arange(len(fft2a)),np.absolute(fft2a), label="Rc("+str(field2)+")",alpha=.5)
    plt.legend()
    plt.xlabel("Harmonic")
    plt.ylabel("JdotE fourier constant")
    plt.title("Fourier components of "+ str(field)+ " and "+ str(field2)+ "T at rho="+ str(x0) + " TE"+str(n)+","+str(m))
    plt.show()
    
    return [fft1a,fft1b], [fft2a,fft2b]
    