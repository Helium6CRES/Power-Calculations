# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 08:45:42 2022

@author: rjtayl
"""

import numpy as np
import scipy.special as sp
from scipy.integrate import simps, trapz
from scipy.optimize import curve_fit
from joblib import Parallel, delayed
import multiprocessing
import pickle
import sys
import time
#import mpmath
from os.path import isfile
import matplotlib.pyplot as plt

num_cores = multiprocessing.cpu_count()

slope_bs = [1,1.25,1.5,2]
slopes = [4126394052.044609, 8698884758.364311, 15613382899.62825, 39144981412.639404]

B_dat = np.array([0.7,1,1.25,1.5,1.7553298,2.0063016,2.2567213, 2.5071204, 2.7579475, 3.0057978])
meanSlope = np.array([3.0e8,3.377e9, 7.61e9, 14.9e9, 25.2e9, 39.7e9, 56.3e9, 80.1e9, 107.1e9, 139.2e9]) #Hz/s
std1 = np.array([0.2, 0.8483, 2.63, 3.47, 5.38, 5.80, 6.23, 6.84, 5.16, 7.43]) #Hz/s

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
    p = np.sqrt(((energy+mc2)**2-mc2**2)/c**2) / JeV
    v = p / (gamma * me)
    vp = v * np.sin(theta*np.pi / 180)
    return (gamma * me * vp)/(e*field)

def velocity(energy, field, theta=90):
    gamma = energy / mc2 +1
    p = np.sqrt(((energy+mc2)**2-mc2**2)/c**2) / JeV
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

#returns modified list of rhos removing values that would hit wall
def max_rho(f, rhos0, field,length):
    rho_max = 0
    for rho in rhos0:
        if circle_traj_data(f, rho, 0, field) != 0:
            rho_max = rho
    return np.linspace(0,rho_max,length)

######## Power Calculation functions ########

# creates perfectly circular trajectory for a given field, center position, and frequency
# outputs velocities, positions, times, and frequencies
# we output frequencies because the mode calculations can also take Kassiopeia trajectories as input, which have varying frequencies
def circle_traj_data(frequency, x0,y0,field,N=1000):
    
    w = 2*np.pi*frequency
    energy = freq_to_energy(frequency, field)
    Rc = cyc_radius(energy, field)
    t = np.linspace(0,1/frequency,N)
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
        return 0
    return [[vx,vy,vz], [x,y,z], t, f]

#general cutoff wavenumbers for TE and TM modes

def bp_0(n,m):
    u = np.int64(4*n**2)
    b =  (m+n/2-3/4)*np.pi
    return b - ((u+3)/(8*b)) - ((4*(7*u**2+82*u-9))/(3*(8*b)**3)) - ((32*(83*u**3 + 2075*u**2 - 3039*u-3537))/(15*(8*b)**5))

def b_0(n,m):
    u = np.int64(4*n**2)
    a = (m+ n/2 - 1/4)*np.pi
    return a - ((u-1)/(8*a)) - ((4*(u-1)*(7*u-31))/(3*(8*a)**3)) - ((32*(u-1)*(83*u**2 - 982*u + 3779))/(15*(8*a)**5))


#load precalculated bessel zeros
with open("jnps.pkl", "rb") as file:
        jnps = pickle.load(file)
        
with open("jns.pkl", "rb") as file:
        jns = pickle.load(file)
        
def jnp(n,m):
    return jnps[n][m]
                     
def jn(n,m):
    return jns[n][m]

def kcTM(n,m,a=a0):
    return jn(n,m)/a

def kcTE(n,m,a=a0):
    return jnp(n,m)/a        
        

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

#Power from larmor formula
def PL(frequency, field, pitch = 90):
    energy = freq_to_energy(frequency, field)
    gamma = energy / mc2 +1
    beta = np.sqrt(1-gamma**-2)
    rc = cyc_radius(energy, field)
    return (2*e**2*c*beta**4*gamma**4)/(12*np.pi*epsilon*rc**2)
    
#Total power that can propogate down guide for TM and TE modes (normalization constant)
def TE_P0(w, n, m):
    return np.pi*w*mu*TEbeta(w, n, m)/2/kcTE(n, m)**4*(jnp(n,m)**2-n**2)*sp.jn(n, jnp(n,m))**2

def TM_P0(w, n, m):
    return np.pi*w*epsilon*TMbeta(w, n, m)/2/kcTM(n, m)**4*(jn(n,m))**2*(sp.jvp(n,jn(n,m)))**2

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
    Ab = -trapz((Exb*Js[0]+Eyb*Js[1]), times)/(PN*times[-1])
    Aa = -trapz((Exa*Js[0]+Eya*Js[1]), times)/(PN*times[-1])
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
    
    Ab = -trapz((Exb*Js[0]+Eyb*Js[1]), times)/(PN*times[-1])
    Aa = -trapz((Exa*Js[0]+Eya*Js[1]), times)/(PN*times[-1])
    return (np.absolute(Ab)**2+np.absolute(Aa)**2)*PN

#calculate total power up to given harmonic
def Pt(field, rho, f,h1,h2):
    P=[]
    trajectory = circle_traj_data(f, rho, 0, field)
    for harmonic in np.arange(h1,h2+1):
        #see report for why I search up to 2h modes
        TEmodes, TMmodes = mode_count(f*harmonic*2*np.pi, harmonic*2)
        # Calculates each mode in parallel to save time, returns list of powers
        pm = Parallel(n_jobs=num_cores)(delayed(TMnm)(*trajectory,mode[0],mode[1],harmonic) for mode in TMmodes)
        pe = Parallel(n_jobs=num_cores)(delayed(TEnm)(*trajectory,mode[0],mode[1],harmonic) for mode in TEmodes)
        
        #add up mode powers and add to total
        P.append(np.sum(pm) + np.sum(pe))
        
        
    return P

def Pt2(field, rho, f,h1, h2):
    P=[]
    trajectory = circle_traj_data(f, rho, 0, field)
    for harmonic in np.arange(h1,h2+1):
        #see report for why I search up to 2h modes
        TEmodes, TMmodes = mode_count(f*harmonic*2*np.pi, harmonic*2)        
        TE_input = [(*trajectory, mode[0],mode[1],harmonic) for mode in TEmodes]
        TM_input = [(*trajectory, mode[0],mode[1],harmonic) for mode in TMmodes]
        
        # Calculates each mode in parallel to save time, returns list of powers
        pool=multiprocessing.Pool()
        pe = pool.starmap(TEnm, TE_input)
        pm = pool.starmap(TEnm, TM_input)
        pool.close()
        
        
        #add up mode powers and add to total
        P.append(np.sum(pm) + np.sum(pe))
    return P

def Pt3(field, rho, f,h1,h2):
    P=0
    
    for harmonic in np.arange(h1,h2+1):
        #see report for why I search up to 2h modes
        TEmodes, TMmodes = mode_count(f*harmonic*2*np.pi, harmonic*2)
        trajectory = circle_traj_data(f, rho, 0, field)
        
        TE_input = [(*trajectory, mode[0],mode[1],harmonic) for mode in TEmodes]
        TM_input = [(*trajectory, mode[0],mode[1],harmonic) for mode in TMmodes]
        
        # Calculates each mode in parallel to save time, returns list of powers
        pool=multiprocessing.Pool()
        pe = pool.starmap_async(TEnm, TE_input)
        pm = pool.starmap_async(TEnm, TM_input)
        
        pe.wait()
        pm.wait()
        
        #add up mode powers and add to total
        P += np.sum(pm.get(timeout=None)) + np.sum(pe.get(timeout=None))
    return P

def Pt4(field, rho, f,h1,h2):
    P=0
    for harmonic in np.arange(h1,h2+1):
        TEmodes, TMmodes = mode_count(f*harmonic*2*np.pi, harmonic*2)
        p=0
        for mode in TMmodes:
            p += (TMnm(*circle_traj_data(f, rho, 0, field),mode[0],mode[1],harmonic)) 
        for mode in TEmodes:
            p += (TEnm(*circle_traj_data(f, rho, 0, field),mode[0],mode[1],harmonic)) 
        P += p
    return P
            
def main():
    #### Load Variables #####
    Field = float(sys.argv[1])
    L_rho = int(sys.argv[2])
    freq = float(sys.argv[3])
    Harmonic= int(sys.argv[4])
    
    #### check for previous solutions then solve ####
    filename = "Pt_" + str(Field) + "T" + str(freq/1e9) + "GHz" + str(Harmonic) + "H"
    rhos = max_rho(freq, np.linspace(0,a0,L_rho), Field, L_rho)

    if not isfile(filename+".pkl"):
        rhos = max_rho(freq,np.linspace(0,a0,L_rho),Field,L_rho)
        Powers = [Pt3(Field,rho,freq,Harmonic,Harmonic) for rho in rhos]
        with open(filename+".pkl", "wb") as file:
            pickle.dump({"rhos":rhos,"powers":Powers},file)
    else:
        print("solution already exists")
    
    return

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    
    starttime = time.time()
    
    main()
    
    endtime = time.time()
    print(endtime-starttime)
