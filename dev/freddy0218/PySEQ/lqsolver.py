"""
Functions for Lindzen-Quo PDE solver for a rectangular domain

@author: itam
"""
import numpy as np
import helper
from scipy import linalg #Replace RMat_Lib.f90


class LQ_solver:
    def __init__(self,Is_Dipole=None,Is_Q=None,Is_MF=None,idealize=False,real_heat=None,
                    id_set=None,VG=None,drm=2000,dzm=1000):
        self.IzMAX = 21
        self.IrMAX = 501
        self.FM = np.zeros((self.IzMAX,self.IrMAX)) #Momentum Forcing
        self.FQ = np.zeros((self.IzMAX,self.IrMAX)) #Diabatic Forcing
        self.R = drm*np.linspace(0,self.IrMAX,self.IrMAX-1)
        self.drm=2000
        self.dzm=1000
        self.fCOR=1e-4
        
        if idealize is True:
            if Is_Dipole is False:
                if Is_MF is True:
                    self.FM[0,:] = -id_set['CD']*VG[0,:]**2/dzm 
                if Is_Q is True:
                    for Iz in range(len(self.IzMAX)):
                        Z = helper.Altitude(1000,Iz)
                        RQ = id_set['RQB']+(id_set['RQT']-id_set['RQB'])*Z/id_set['ZQT']
                        Z_FAC=helper.weight((Z-id_set['ZQM'])/(id_set['ZQT']-id_set['ZQM']))
                        for Ir in range(len(self.IrMAX)):
                            self.FQ[Iz,Ir] = id_set['AQ']*Z_FAC*helper.hump((helper.Radius(2000,Ir)-RQ)/id_set['SQ'])
            else:
                if Is_MF is True:
                    for Iz in range(self.IzMAX):
                        for Ir in range(self.IrMAX):
                            ARG = np.sqrt((helper.Radius(2000,Ir)-id_set["RMD"])**2+(helper.Altitude(1000,Iz)-id_set["ZMD"])**2)/id_set["DZQD"]
                            self.FM[Iz,Ir]=id_set["FMD"]*helper.hump(ARG)
                if Is_Q is True:
                    for Iz in range(self.IzMAX):
                        for Ir in range(self.IrMAX):
                            ARG = np.sqrt((helper.Radius(2000,Ir)-id_set["RQD"])**2+(helper.Altitude(1000,Iz)-id_set["ZQD"])**2)/id_set["DRQD"]
                            self.FQ[Iz,Ir]=id_set["FQD"]*helper.hump(ARG)
        else:
            if Is_Dipole is False:
                if Is_MF is True:
                    self.FM[0,:] = -id_set['CD']*VG[0,:]**2/dzm 
                if Is_Q is True:
                    self.FQ[:] = real_heat[:]
                    print("Successfully reading in the real heating structure!")
    
    def make_dsc(self,N2=None,I2=None,B2=None,gamma=None):
        """

        Parameters
        ----------
        N2 : TYPE, optional
            Buoyancy Frequency
        I2 : TYPE, optional
            Baroclinicity
        B2 : TYPE, optional
            Local Intertia Frequency
        gamma : TYPE, optional
            Slope of Isobaric surfaces

        Returns
        -------
        DSC (4A*C-B2 discriminant for second-order PDE)

        """
        DSC = np.zeros((self.IzMAX,self.IrMAX))
        for Iz in range(self.IzMAX):
            for Ir in range(self.IrMAX):
                DSC[Iz,Ir] = 4*(N2[Iz,Ir]*(I2[Iz,Ir]-gamma(Iz,Ir)*B2[Iz,Ir])-B2[Iz,Ir]**2)
        return DSC
                    
    def make_uvw(self,Y=None,VG=None,RHO=None):
        #....................
        # Parameters
        #....................
        IrM1 = self.IrMAX-1
        IrM2 = IrM1-1
        IzM1 = self.IzMAX-1
        nn=self.IzMAX
        
        #------------------------------
        # Populate zero arrays
        #------------------------------
        UU = np.zeros((self.IzMAX,self.IrMAX)) #Radial wind
        WW = np.zeros((self.IzMAX,self.IrMAX)) #Vertical wind
        Vv = np.zeros((self.IzMAX,self.IrMAX)) #Swirling wind
        
        # Initialize extreme wind components
        Umx=-999.9
        Umn=999.9
        Vmx=-999.9
        Vmn=999.9
        Wmx=-999.9
        Wmn=999.9
        
        for Ir in range(1,IrM2):
            # Top and bottom boundaries
            UU[0,Ir] = -(Y[1,Ir]-Y[0,Ir])/(RHO[0,Ir]*helper.Rm(2000,Ir)*self.dzm)
            Umx,Umn = helper.XTEME(Umx,Umn,UU[0,Ir])
            UU[-1,Ir]=-(Y[-1,Ir]-Y[IzM1-1,Ir])/(RHO[-1,Ir]*helper.Rm(2000,Ir)*self.dzm)
            Umx,Umn = helper.XTEME(Umx,Umn,UU[-1,Ir])
            # No vertical flow across boundaries
            WW[0,:] = 0.0
            WW[-1,:] = 0.0
            for Iz in range(1,IzM1):
                #Radial flow in interior points from streamfunction
                UU[Iz,Ir] = -(Y[Iz+1,Ir]-Y[Iz-1,Ir])/(2*RHO[Iz,Ir]*helper.Rm(2000,Ir)*self.dzm)
                Umx,Umn = helper.XTEME(Umx,Umn,UU[Iz,Ir])
                #Vertical flow in interior points
                WW[Iz,Ir] = (Y[Iz,Ir+1]-Y[Iz,Ir-1])/(2*RHO[Iz,Ir]*helper.Rm(2000,Ir)*self.drm)
                Wmx,Wmn = helper.XTEME(Wmx,Wmn,WW[Iz,Ir])
                
        # Loop vertically at center and vortex edge
        for Iz in range(0,self.IzMAX):
            UU[Iz,0] = 0.0
            UU[Iz,IrM1-1] = UU[Iz,IrM2]
            UU[Iz,self.IrMAX-1] = UU[Iz,IrM1]
            
            WW[Iz,0] = Y[Iz,1]/(RHO[Iz,0]*helper.Rm(2000,1)*self.drm)
            Wmx,Wmn = helper.XTEME(Wmx,Wmn,WW[Iz,0])
            
            WW[Iz,IrM1] = 0.0
            WW[Iz,self.IrMAX] = 0.0
            
        for Ir in range(1,IrM2):
            E0 = 2*VG[0,Ir]/helper.Rm(2000,Ir)+self.fCOR
            AKEm = 0.5*UU[0,Ir-1]**2
            AKEp = 0.5*UU[0,Ir+1]**2
            Vv[0,Ir] = (AKEp-AKEm)/(2*self.drm*E0)
            
            E0 = 2*VG[self.IzMAX-1,Ir]/helper.Rm(2000,Ir)+self.fCOR
            AKEm = 0.5*UU[self.IzMAX-1,Ir-1]**2
            AKEp = 0.5*UU[self.IzMAX-1,Ir+1]**2
            Vv[self.IzMAX-1,Ir]=(AKEp-AKEm)/(2.0*self.drm*E0)
            for Iz in range(1,IzM1):
                E0 = 2*VG[Iz,Ir]/helper.Rm(2000,Ir)+self.fCOR
                AKEm = 0.5*(UU[Iz,Ir-1]**2+WW[Iz,Ir-1]**2)
                AKEp = 0.5*(UU[Iz,Ir+1]**2+WW[Iz,Ir+1]**2)
                Z_L = (WW[Iz,Ir+1]-WW[Iz,Ir-1])/(2*self.drm)-(UU[Iz+1,Ir]-WW[Iz-1,Ir])/(2*self.dzm)
                Vv[Iz,Ir]=((AKEp-AKEm)/(2*self.drm)-WW[Iz,Ir]*Z_L/E0)
                Vmx,Vmn = helper.XTEME(Vmx,Vmn,Vv[Iz,Ir])
                
        for Iz in range(0,self.IzMAX):
            Vv[Iz,0] = 0.0
            Vv[Iz,self.IrMAX-1] = Vv[Iz,IrM1-1]
            
        return UU,WW,Vv
    
    
            
            
            
        
                    
                            
                            
                    
                    
                    
                        
        
        
        
        
