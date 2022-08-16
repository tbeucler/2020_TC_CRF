#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 15:14:37 2021

@author: itam
"""

import numpy as np
import helper

IzMAX = 21
IrMAX = 501
dzm=1000
drm=2000
Cp=1004
GoverCp = 0.009761

def Mak_NBI2(Bo=None,fCOR=1e-4,VG=None):
    # Computes B-V frequency, baroclinicity, inertia frequency, gradient acceleration
    # divided by gravity <-- gradient wind, balanced buoyancy, coriolis parameter
    
    N2 = np.zeros((IzMAX,IrMAX))
    I2 = np.zeros((IzMAX,IrMAX))
    B2 = np.zeros((IzMAX,IrMAX))
    gamma = np.zeros((IzMAX,IrMAX))
    
    for Ir in range(IrMAX):
        N2[0,Ir] = (Bo[1,Ir]-Bo[0,Ir])/dzm
        N2[-1,Ir] = (Bo[-1,Ir]-Bo[-2,Ir])/dzm
        for Iz in range(1,IzMAX-1):
            N2[Iz,Ir] = (Bo[Iz+1,Ir]-Bo[Iz-1,Ir])/(2*dzm)
    
    for Ir in range(1,IrMAX-1):
        R = helper.Rm(2000,Ir)
        for Iz in range(IzMAX):
            B2[Iz,Ir] = (Bo[Iz,Ir+1]-Bo[Iz,Ir-1])/(2*drm)
            vor = (VG[Iz,Ir+1]-VG[Iz,Ir-1])/(2*drm)+VG[Iz,Ir]/R+fCOR
            I2[Iz,Ir] = vor*(2*VG[Iz,Ir]/R+fCOR)
            
    R=helper.Rm(2000,IrMAX)
    for Iz in range(IzMAX):
        B2[Iz,-1] = (Bo[Iz,-1]-Bo[Iz,-2])/drm
        B2[Iz,0] = 0.0
        I2[Iz,0] = (2*VG[Iz,1]/drm+fCOR)**2
        vor = (VG[Iz,-1]-VG[Iz,-2])/drm+VG[Iz,-1]/R+fCOR
        I2[Iz,-1] = vor*(2*VG[Iz,-1]/R+fCOR)
        
    for Ir in range(1,IrMAX):
        R=helper.Rm(2000,Ir)
        for Iz in range(IzMAX):
            gamma[Iz,Ir] = (1/9.8)*VG[Iz,Ir]*(VG[Iz,Ir]/R+fCOR)
    for Iz in range(IzMAX):
        gamma[Iz,0] = 0.0
        
    return N2,I2,B2,gamma


def MakPI(PI0=None,VG=None,TH=None,fCOR=1e-4):
    # Integrates gradient wind relation inward from vortex periphery -> Exner function in balance with wind
    PI = np.zeros_like(VG)
    for Iz in range(1,IzMAX-1):
        sum8 = 0.0
        for Ir in reversed(range(1,IrMAX-1)):
            RA = helper.Rm(2000,Ir+1)
            RAM = RA-drm
            sum8 = sum8+float(0.5*drm*(VG[Iz,Ir+1]*(VG[Iz,Ir+1]/RA+fCOR)/(Cp*TH[Iz,Ir+1])+\
                                       VG[Iz,Ir]*(VG[Iz,Ir]/RAM+fCOR)/(Cp*TH[Iz,Ir])))
            PI[Iz,Ir] = PI0[Iz]-np.real(sum8)
    return PI

PI0 = np.random.rand(21)
VG = np.random.rand(21,501)
TH = np.random.rand(21,501)

ans = MakPI(PI0,VG,TH,1e-4)

            
            
            
            
    