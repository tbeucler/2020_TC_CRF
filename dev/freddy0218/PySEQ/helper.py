#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 18:41:14 2021

@author: itam
"""

import numpy as np

def weight(X=None):
    """
    For a normalized argument X, returns a ramp function with C2 continuity. 
    X=0 -> output=1; X=1 -> output=0
    """
    if X<0:
        weight=1
    elif X>1:
        weight=0
    else:
        weight=1-(10-(15-6*X)*X)*X*X*X
    return weight

def hump(X=None):
    """
    Fpr a normalized argument X, returns a hump function with C2 continuity.
    Output=1 (X=0); Output=0(X=+-1) 

    """
    if np.abs(X)<1:
        x2 = X*X
        hump = 1-x2*(2-x2)
    else:
        hump = 0
    return hump

def Altitude(dz=1000,Iz=None):
    """
    Vertical index to altitude
    """
    return dz*(Iz-1)

def Radius(dr=2000,Ir=None):
    return dr*(Ir-1)

def Rm(dr=2000,Ir=None):
    return dr*(Ir-1)

def Zm(dz=1000,Iz=None):
    return dz*(Iz-1)

def XTEME(Xmax=None,Xmin=None,Xcandidate=None):
    # Replace former max/min values with current value if current value is more extreme
    Xmx=max(Xmax,Xcandidate)
    Xmn=min(Xmin,Xcandidate)
    return Xmx,Xmn
