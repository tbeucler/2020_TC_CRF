import xarray as xr
import numpy as np
from tools import read_and_proc
from scipy.ndimage import gaussian_filter
import gc,glob
from tqdm import tqdm
import dask.array as da
from dask import delayed

def dummy(inlist=None):
    return inlist

class preprocess:
    def __init__(self,uvtpath=None,originpath=None,expname=None,window=None,addctrlmethod='orig',gaussian=True,sigma=None,surfix=None):
        self.uvtpath=uvtpath
        self.originpath=originpath
        self.expname=expname
        self.window=window
        self.method=addctrlmethod
        self.gaussian=gaussian
        self.sigma=sigma
        self.surfix=surfix
        self.coor = None#xr.open_dataset('/scratch/06040/tg853394/tc/output/redux/maria/ctl/post/U.nc')
    #....................................................................................................................................
    # Helper functions
    #....................................................................................................................................
    def nearest_index(self,array=None,value=None):
        idx = (np.abs(array-value)).argmin()
        return idx
    
    def swap_presaved(self,varname=None,ctl=True):
        if ctl is True:
            return np.swapaxes(np.asarray(read_and_proc.depickle(self.uvtpath+'ctl'+'/'+varname)),0,1)
        else:
            return np.swapaxes(np.asarray(read_and_proc.depickle(self.uvtpath+str(self.expname)+'/'+varname)),0,1)
    
    def read_azimuth_fields(self,varname=None,wantR=False,ctl=True):
        if ctl is True:
            temp = read_and_proc.read_some_azimuth_fields(fileloc=[glob.glob(self.originpath+'ctl'+'/azim_'+str(varname)+'_*')[0]],fieldname=[varname])
        else:
            temp = read_and_proc.read_some_azimuth_fields(fileloc=[glob.glob(self.originpath+str(self.expname)+'/azim_'+str(varname)+'_*')[0]],fieldname=[varname])
            
        if wantR is True:
            return temp[varname][varname],self.nearest_index(temp[varname][varname].radius,500)
        else:
            return temp[varname][varname]
        
    def stickCTRL(self,ctrlvar=None,senvar=None,method='orig'):
        if method=='orig':
            return read_and_proc.add_ctrl_before_senstart(ctrlvar,senvar,self.expname,'Yes')
        else:
            return read_and_proc.add_ctrl_before_senstart_ctrlbase(ctrlvar,senvar,self.expname,'Yes')
        
    def smooth_to_dict(self,var=None,varname=None,window=None):
        ass2 = {}
        for indx,name in enumerate(varname):
            if self.gaussian:
                temp = delayed(self.smooth_array_with_gaussian)(var[indx],0,self.sigma)
            else:
                temp = delayed(self.smooth_array_along_axis)(var[indx],0,window)
            ass2[name] = temp
        assmodel = delayed(dummy)(ass2)
        outdict = assmodel.compute()
        return outdict
    
    def smooth(self,varlist=None):
        return [gaussian_filter(var,sigma=self.sigma) for var in varlist]
    
    def to_dict(self,var=None,varname=None):
        outdict = {}
        for indx,name in enumerate(varname):
            outdict[name] = var[indx]
        return outdict
    
    def smooth_array_along_axis(self,array=None,axis=None,window=3):
        if axis==0:
            aN = array.copy()
        else:
            aN = np.swapaxes(array,axis,0)
        aN = aN.reshape((aN.shape[0], -1))
        return np.asarray([np.convolve(aN[:,i],np.ones(window)/window,'same')[0:aN.shape[0]] for i in range(len(aN[0,:]))]).transpose()
    
    def smooth_array_with_gaussian(self,array=None,axis=None,sigma=[3,0,0,0]):
        if axis==0:
            aN = array.copy()
        else:
            aN = np.swapaxes(array,axis,0)
        temp = gaussian_filter(aN,sigma=sigma)
        return temp.reshape((temp.shape[0], -1))
    
    def forward_diff(self,arrayin=None,delta=None,axis=None,LT=1):
        result = []
        if axis==0:
            for i in range(0,arrayin.shape[axis]-LT):
                temp = (arrayin[i+LT,:]-arrayin[i,:])/(LT*delta)
                result.append(temp)
            return np.asarray(result)

    #....................................................................................................................................
    # Preprocess
    #....................................................................................................................................
    def preproc_uvwF(self,smooth=True):
        if self.expname=='ctl':
            TYPE=True
        else:
            TYPE=False
        # Read vars
        urad,vtan=da.from_array(self.swap_presaved('urad',TYPE)[:,:,:,0:167]),da.from_array(self.swap_presaved('vtan',TYPE)[:,:,:,0:167])
        theta=da.from_array(self.swap_presaved('theta',TYPE)[:,:,:,0:167])
        w,r500=self.read_azimuth_fields('W',True,TYPE)
        qv,hdia,rthratlw,rthratsw=da.from_array(self.read_azimuth_fields('QVAPOR',False,TYPE).data)[:,:,:,0:167],\
        self.read_azimuth_fields('H_DIABATIC',False,TYPE),\
        self.read_azimuth_fields('RTHRATLW',False,TYPE),self.read_azimuth_fields('RTHRATSW',False,TYPE)
        # Heat forcing sum
        heatsum = hdia+rthratlw+rthratsw
        del hdia,rthratlw,rthratsw
        gc.collect()
        
        wr,heatsumr = da.from_array(w[:,:,:,0:167]),da.from_array(heatsum[:,:,:,0:167])
        del w,heatsum
        gc.collect()
        
        if self.expname=='ctl':
            outdict=self.smooth_to_dict([urad,vtan,wr,qv,theta,heatsumr],['u','v','w','qv','theta','heatsum'],self.window)
            del urad,vtan,qv,wr,heatsumr 
            gc.collect()
            
            folderpath='/work/FAC/FGSE/IDYST/tbeucler/default/freddy0218/TCGphy/2020_TC_CRF/dev/freddy0218/'
            if smooth is True:
                read_and_proc.save_to_pickle(folderpath+'pca/output/uvwheat/'+str(self.expname)+'_smooth_'+self.surfix,outdict,'PICKLE')
            else:
                read_and_proc.save_to_pickle(folderpath+'pca/output/uvwheat/'+str(self.expname)+'_'+self.surfix,outdict,'PICKLE')                
            print("---Finish!---")
            return outdict,r500
        else:
            uradC,vtanC,thetaC=da.from_array(self.swap_presaved('urad',True)[:,:,:,0:167]),da.from_array(self.swap_presaved('vtan',True)[:,:,:,0:167]),da.from_array(self.swap_presaved('theta',True)[:,:,:,0:167])
            wC,qvC,hdiaC,rthratlwC,rthratswC=self.read_azimuth_fields('W',False,True),\
            da.from_array(self.read_azimuth_fields('QVAPOR',False,True)[:,:,:,0:167]),\
            self.read_azimuth_fields('H_DIABATIC',False,True),self.read_azimuth_fields('RTHRATLW',False,True),self.read_azimuth_fields('RTHRATSW',False,True)
            heatsumC=hdiaC+rthratlwC+rthratswC
            del hdiaC,rthratlwC,rthratswC
            gc.collect()
            
            wCr,heatsumCr = da.from_array(wC[:,:,:,0:167]),da.from_array(heatsumC[:,:,:,0:167])
            del wC,heatsumC
            gc.collect()
            
            uradL,vtanL,wL,qvL,thetaL,heatsumL=da.from_array(np.asarray(self.stickCTRL(uradC,urad,self.method))),\
            da.from_array(np.asarray(self.stickCTRL(vtanC,vtan,self.method))),\
            da.from_array(np.asarray(self.stickCTRL(wCr,wr,self.method))),\
            da.from_array(np.asarray(self.stickCTRL(qvC,qv,self.method))),\
            da.from_array(np.asarray(self.stickCTRL(thetaC,theta,self.method))),\
            da.from_array(np.asarray(self.stickCTRL(heatsumCr,heatsumr,self.method)))
            del uradC,urad,theta,vtanC,thetaC,vtan,wCr,wr,qvC,qv,heatsumCr,heatsumr
            gc.collect()
                
            outdict=self.smooth_to_dict([uradL,vtanL,wL,qvL,thetaL,heatsumL],['u','v','w','qv','theta','heatsum'],self.window)
            del uradL,vtanL,wL,qvL,heatsumL
            gc.collect()
            print("---Finish!---")
            
            folderpath='/work/FAC/FGSE/IDYST/tbeucler/default/freddy0218/TCGphy/2020_TC_CRF/dev/freddy0218/'
            if smooth is True:
                read_and_proc.save_to_pickle(folderpath+'pca/output/uvwheat/'+str(self.expname)+'_smooth_'+self.surfix,outdict,'PICKLE')
            else:
                read_and_proc.save_to_pickle(folderpath+'pca/output/uvwheat/'+str(self.expname)+'_'+self.surfix,outdict,'PICKLE')                
            del outdict
            gc.collect()
            return None

    def preproc_onevar(self,varname='W',TYPEsmooth='orig',save_postfix=None):
        if self.expname=='ctl':
            TYPE=True
        else:
            TYPE=False
            
        if varname=='theta':
            qv=da.from_array(self.swap_presaved('theta',TYPE)[:,:,:,0:167])
        else:
            qv=da.from_array(self.read_azimuth_fields(varname,False,True)[:,:,:,0:167])
        
        if self.expname=='ctl':
            outdict=self.smooth_to_dict([qv],[varname],self.window)
            del qv
            gc.collect()
            print("---Finish!---")
            return outdict
        else:
            if varname=='theta':
                qvC=da.from_array(self.swap_presaved('theta',True)[:,:,:,0:167])
            else:
                qvC=da.from_array(self.read_azimuth_fields(varname,False,True)[:,:,:,0:167])
            if TYPEsmooth=='orig':
                qvL=da.from_array(np.asarray(self.stickCTRL(qvC,qv,self.method)))
                del qvC,qv
                gc.collect()
                
                outdict=self.smooth_to_dict([qvL],[varname],self.window)
                del qvL
                gc.collect()
                print("---Finish!---")
                return outdict
            
    def preproc_dudvdw(self):
        if self.expname=='ctl':
            TYPE=True
        else:
            TYPE=False
        urad,vtan=da.from_array(self.swap_presaved('urad',TYPE)[:,:,:,0:167]),da.from_array(self.swap_presaved('vtan',TYPE)[:,:,:,0:167])
        w=da.from_array(self.read_azimuth_fields('W',True,TYPE)[0][:,:,:,0:167])
        if self.expname=='ctl':
            outdict=self.smooth_to_dict([urad,vtan,w],['u','v','w'],self.window)
            durad,dvtan,dw=self.forward_diff(arrayin=outdict['u'],delta=60*60,axis=0),self.forward_diff(arrayin=outdict['v'],delta=60*60,axis=0),\
            self.forward_diff(arrayin=outdict['w'],delta=60*60,axis=0)
            print(durad.shape)
        else:
            uradC,vtanC=da.from_array(self.swap_presaved('urad',True)[:,:,:,0:167]),da.from_array(self.swap_presaved('vtan',True)[:,:,:,0:167])
            wC=self.read_azimuth_fields('W',True,True)[0][:,:,:,0:167]
            uradL,vtanL,wL = da.from_array(np.asarray(self.stickCTRL(uradC,urad,self.method))),da.from_array(np.asarray(self.stickCTRL(vtanC,vtan,self.method))),\
            da.from_array(np.asarray(self.stickCTRL(wC,w,self.method)))
            
            outdict=self.smooth_to_dict([uradL,vtanL,wL],['u','v','w'],self.window)
            durad,dvtan,dw=self.forward_diff(arrayin=outdict['u'],delta=60*60,axis=0),self.forward_diff(arrayin=outdict['v'],delta=60*60,axis=0),\
            self.forward_diff(arrayin=outdict['w'],delta=60*60,axis=0)
            del uradC,vtanC,wC,urad,vtan,w,uradL,vtanL,wL
            gc.collect()
        durad = np.nan_to_num(durad)
        dvtan = np.nan_to_num(dvtan)
        dw = np.nan_to_num(dw)
        outdictuvw = {'du':durad,'dv':dvtan,'dw':dw}
        print("---Finish!---")
        return outdictuvw
    
    def preproc_uvw(self):
        if self.expname=='ctl':
            TYPE=True
        else:
            TYPE=False
        urad,vtan=da.from_array(self.swap_presaved('urad',TYPE)[:,:,:,0:167]),da.from_array(self.swap_presaved('vtan',TYPE)[:,:,:,0:167])
        w=da.from_array(self.read_azimuth_fields('W',True,TYPE)[0][:,:,:,0:167])
        if self.expname=='ctl':
            outdict=self.smooth_to_dict([urad,vtan,w],['u','v','w'],self.window)
        else:
            uradC,vtanC=da.from_array(self.swap_presaved('urad',True)[:,:,:,0:167]),da.from_array(self.swap_presaved('vtan',True)[:,:,:,0:167])
            wC=self.read_azimuth_fields('W',True,True)[0][:,:,:,0:167]
            uradL,vtanL,wL = da.from_array(np.asarray(self.stickCTRL(uradC,urad,self.method))),da.from_array(np.asarray(self.stickCTRL(vtanC,vtan,self.method))),\
            da.from_array(np.asarray(self.stickCTRL(wC,w,self.method)))
            
            outdict=self.smooth_to_dict([uradL,vtanL,wL],['u','v','w'],self.window)
            del uradC,vtanC,wC,urad,vtan,w,uradL,vtanL,wL
            gc.collect()
            
        outdictuvw = {'u':outdict['u'],'v':outdict['v'],'w':outdict['w']}
        print("---Finish!---")
        return outdictuvw
    
    def preproc_dthdQ(self):
        if self.expname=='ctl':
            TYPE=True
        else:
            TYPE=False
        theta=da.from_array(self.swap_presaved('theta',TYPE)[:,:,:,0:167])
        hdia,rthratlw,rthratsw=self.read_azimuth_fields('H_DIABATIC',False,TYPE),self.read_azimuth_fields('RTHRATLW',False,TYPE),self.read_azimuth_fields('RTHRATSW',False,TYPE)
        heatsum = hdia+rthratlw+rthratsw
        del hdia,rthratlw,rthratsw
        gc.collect()
        
        dQ = da.from_array(heatsum[:,:,:,0:167])
        del heatsum
        gc.collect()
        
        if self.expname=='ctl':
            outdict=self.smooth_to_dict([theta,dQ],['theta','heatsum'],self.window)
            dth,dQ=self.forward_diff(arrayin=outdict['theta'],delta=60*60,axis=0),self.forward_diff(arrayin=outdict['heatsum'],delta=60*60,axis=0)
            print(dth.shape)
        else:
            thetaC=da.from_array(self.swap_presaved('theta',True)[:,:,:,0:167])
            hdiaC,rthratlwC,rthratswC=self.read_azimuth_fields('H_DIABATIC',False,True),self.read_azimuth_fields('RTHRATLW',False,True),self.read_azimuth_fields('RTHRATSW',False,True)
            heatsumC=hdiaC+rthratlwC+rthratswC
            del hdiaC,rthratlwC,rthratswC
            gc.collect()
            
            dQC = da.from_array(heatsumC[:,:,:,0:167])
            del heatsumC
            gc.collect()
            
            thetaL,dQL = da.from_array(np.asarray(self.stickCTRL(thetaC,theta,self.method))),da.from_array(np.asarray(self.stickCTRL(dQC,dQ,self.method)))
            
            outdict=self.smooth_to_dict([thetaL,dQL],['theta','heatsum'],self.window)
            dth,dQ=self.forward_diff(arrayin=outdict['theta'],delta=60*60,axis=0),self.forward_diff(arrayin=outdict['heatsum'],delta=60*60,axis=0)
            del thetaL,dQL
            gc.collect()
        dth = np.nan_to_num(dth)
        dQ = np.nan_to_num(dQ)
        outdictthQ = {'dth':dth,'dQ':dQ}
        print("---Finish!---")
        return outdictthQ
    
    def preproc_heateq(self,outputVAR=None):
        if self.expname=='ctl':
            TYPE=True
        else:
            TYPE=False
            
        if outputVAR=='IR':
            hdia = self.read_azimuth_fields('H_DIABATIC',False,TYPE)
            rthratlw,rthratsw = self.read_azimuth_fields('RTHRATLW',False,TYPE),self.read_azimuth_fields('RTHRATSW',False,TYPE)
            rthratlwc,rthratswc = self.read_azimuth_fields('RTHRATLWC',False,TYPE),self.read_azimuth_fields('RTHRATSWC',False,TYPE)
            ir = (rthratlw-rthratlwc)+(rthratsw-rthratswc)
            noir = hdia+rthratlw+rthratsw-ir
            dQ,dNQ = da.from_array(ir[:,:,:,0:167]),da.from_array(noir[:,:,:,0:167])
            del hdia,rthratlw,rthratsw,rthratswc,rthratlwc,ir,noir
            gc.collect()
        elif outputVAR=='RAD':
            hdia,rthratlw,rthratsw=self.read_azimuth_fields('H_DIABATIC',False,TYPE),self.read_azimuth_fields('RTHRATLW',False,TYPE),self.read_azimuth_fields('RTHRATSW',False,TYPE)
            rad = rthratlw+rthratsw
            norad = hdia
            dQ,dNQ = da.from_array(rad[:,:,:,0:167]),da.from_array(norad[:,:,:,0:167])
            del hdia,rthratlw,rthratsw
            gc.collect()
        
        if self.expname=='ctl':
            outdict=self.smooth_to_dict([dQ,dNQ],['HEAT','NOHEAT'],self.window)
            del dQ,dNQ
            gc.collect()
            print("---Finish!---")
            return outdict
        else:
            if outputVAR=='IR':
                hdiaC,rthratlwC,rthratswC=self.read_azimuth_fields('H_DIABATIC',False,True),self.read_azimuth_fields('RTHRATLW',False,True),self.read_azimuth_fields('RTHRATSW',False,True)
                rthratlwcC,rthratswcC=self.read_azimuth_fields('RTHRATLWC',False,True),self.read_azimuth_fields('RTHRATSWC',False,True)
                irC = (rthratlwC-rthratlwcC)+(rthratswC-rthratswcC)
                noirC = (hdiaC+rthratlwC+rthratswC)-irC
                del hdiaC,rthratlwC,rthratswC,rthratlwcC,rthratswcC
                gc.collect()
                
                dQC,dNQC = da.from_array(irC[:,:,:,0:167]),da.from_array(noirC[:,:,:,0:167])
                del irC,noirC
                gc.collect()
                
            elif outputVAR=='RAD':
                hdiaC,rthratlwC,rthratswC=self.read_azimuth_fields('H_DIABATIC',False,True),self.read_azimuth_fields('RTHRATLW',False,True),self.read_azimuth_fields('RTHRATSW',False,True)
                radC = rthratlwC+rthratswC
                noradC = hdiaC
                del hdiaC,rthratlwC,rthratswC
                gc.collect()
                
                dQC,dNQC = da.from_array(radC[:,:,:,0:167]),da.from_array(noradC[:,:,:,0:167])
                del radC,noradC
                gc.collect()
            
            dQL,dNQL = da.from_array(np.asarray(self.stickCTRL(dQC,dQ,self.method))),da.from_array(np.asarray(self.stickCTRL(dNQC,dNQ,self.method)))
            outdict=self.smooth_to_dict([dQL,dNQL],['HEAT','NOHEAT'],self.window)
            del dQL,dNQL
            gc.collect()
            print("---Finish!---")
            return outdict