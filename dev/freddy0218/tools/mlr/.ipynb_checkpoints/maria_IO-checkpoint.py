import numpy as np
import gc
from tools import read_and_proc

def flatten(t):
    #https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-a-list-of-lists
    return [item for sublist in t for item in sublist]

def prepare_inputdataset2(inputTS=None,flat_out=np.zeros((83,12+3+3+9)),lefttimelim=None):
    def flattenlist(inlist=None):
        tempvarlist = []
        for item in inlist:
            tempvarlist.append(item)
        return read_and_proc.flatten(tempvarlist)
    input_dataset36 = []
    for timeseries in inputTS:#[pca_timeseries_36,pcaur_timeseries_36,pcavr_timeseries_36,pcaw_timeseries_36]:
        input_dataset36.append(timeseries)
    for i in (range(inputTS[0][:,0].shape[0])):
        tempinlist = [obj[i,:] for obj in input_dataset36]
        flat_out[i,:] = flattenlist(tempinlist)
    del timeseries,i
    return flat_out

def forward_diff(arrayin=None,delta=None,axis=None,LT=1):
    result = []
    if axis==0:
        for i in range(0,arrayin.shape[axis]-LT):
            temp = (arrayin[i+LT,:]-arrayin[i,:])/(LT*delta)
            result.append(temp)
        return np.asarray(result)
    
def myPCA_projection_sen(pca_dict=None,varname=None,toproj_flatvar=None,orig_flatvar=None):
    pca_orig = pca_dict[varname].transform(orig_flatvar)
    if pca_dict[varname].mean_ is not None:
        orig_mean = pca_dict[varname].mean_
    projvar_transformed = np.dot(toproj_flatvar-np.nanmean(orig_flatvar,axis=0),pca_dict[varname].components_.T)
    del orig_mean
    gc.collect()
    return pca_orig, projvar_transformed
    
class input_output:
    def __init__(self,PCAdict=None,folderpath=None,ts_varname=None,nummem=None):
        self.PCAdict = PCAdict
        self.expname=['ctl','ncrf_36h','ncrf_60h','lwcrf']
        self.nummem = nummem # u: 36 (40% variability in du), v:16/32 (40% dv var;50%), w:44 (40% dw var)
        self.ts_varname = ts_varname
    ###################################################################################################################################################
    # Read var
    ###################################################################################################################################################
    def readvar(self,listdict=None,varname=['u','v','w','qv','heatsum'],withtheta='Yes',thetaflat=None,smooth24=False):
        # Smooth24 or nosmooth
        vardict = {}
        if smooth24:
            withtheta='No'
            varname=['u','v','w','qv','theta','heatsum']
        else:
            withtheta='Yes'
            varname=['u','v','w','qv','heatsum']
        # Read array
        for indx,obj in enumerate(self.expname):
            if withtheta=='Yes':
                templist = [listdict[indx][strvar][24:120] for strvar in varname]
                theta = thetaflat[list(thetaflat.keys())[indx]][24:120]
                heatsum = templist[4]
                templist.pop(4)
                templist.insert(4,theta)
                templist.insert(5,heatsum)
                assert len(templist)==6
                vardict[obj] = templist
            else:
                vardict[obj] = [listdict[indx][strvar][24:120] for strvar in varname]
        return vardict

    def readvar_sen(self,listdict=None,varname=['u','v','w','qv','heatsum'],withtheta='Yes',thetaflat=None,heatcompflat=None):
        vardict = {}
        for indx,obj in enumerate(self.expname):
            if withtheta=='Yes':
                templist = [listdict[indx][strvar][24:120] for strvar in varname]
                theta = thetaflat[list(thetaflat.keys())[indx]][24:120]
                heatcomp = heatcompflat[list(heatcompflat.keys())[indx]][24:120]
                heatsum = templist[4]
                templist.pop(4)
                templist.insert(4,theta)
                templist.insert(5,heatsum)
                templist.insert(6,heatcomp)
                assert len(templist)==7
                vardict[obj] = templist
        return vardict
    
    ###################################################################################################################################################
    # Produce time series
    ###################################################################################################################################################    
    def produce_timeseries(self,flatvar=None):
        ts_dict = {}
        for indx,obj in enumerate(self.ts_varname):
            flatvarstemp = [flatvar[expNAME][indx] for expNAME in self.expname]
            temp = [self.PCAdict[obj].transform(flatvartempTEMP)[:,0:self.nummem[indx]] for flatvartempTEMP in flatvarstemp]
            del flatvarstemp
            gc.collect()
            ts_dict[obj] = temp
        return ts_dict
    
    def produce_sentimeseries(self,flatvar=None):
        ts_dict = {}
        for indx,obj in enumerate(self.ts_varname):
            if obj != 'heatcomp':
                flatvarstemp = [flatvar[expNAME][indx] for expNAME in self.expname]
                temp = [self.PCAdict[obj].transform(flatvartempTEMP)[:,0:self.nummem[indx]] for flatvartempTEMP in flatvarstemp]
                del flatvarstemp
                gc.collect()
                ts_dict[obj] = temp
            else:
                print(self.nummem[indx])
                heatcomptemp = [flatvar[expNAME][indx] for expNAME in self.expname]
                temp = [myPCA_projection_sen(pca_dict=self.PCAdict,varname='heatsum',toproj_flatvar=heatcomptemp[i],\
                                             orig_flatvar=heatcomptemp[0])[1][:,0:self.nummem[indx]] for i in range(len(heatcomptemp))]
                del heatcomptemp
                gc.collect()
                ts_dict[obj] = temp
        return ts_dict
    
    ###################################################################################################################################################
    # Normalization
    ###################################################################################################################################################     
    def normalize(self,ts=None,TYPE='meanstd'):
        normalize_ts_dict = {}
        for indx,obj in enumerate(self.ts_varname):
            tstemp = [ts[obj][expINDEX] for expINDEX in range(len(self.expname))]
            temp = [(tstemp[i]-np.nanmean(tstemp[0],axis=0))/np.nanstd(tstemp[0],axis=0) for i in range(len(tstemp))]
            normalize_ts_dict[obj] = temp
        return normalize_ts_dict
    
    def normalize_sen(self,ts=None):
        normalize_ts_dict = {}
        for indx,obj in enumerate(self.ts_varname):
            if obj != 'heatsum':
                tstemp = [ts[obj][expINDEX] for expINDEX in range(len(self.expname))]
                temp = [(tstemp[i]-np.nanmean(tstemp[0],axis=0))/np.nanstd(tstemp[0],axis=0) for i in range(len(tstemp))]
                normalize_ts_dict[obj] = temp
            else:
                tstemp = [ts['heatcomp'][expINDEX] for expINDEX in range(len(self.expname))]
                tsbase = [ts['heatsum'][expINDEX] for expINDEX in range(len(self.expname))]
                temp = [(tstemp[i]-np.nanmean(tsbase[0],axis=0))/np.nanstd(tsbase[0],axis=0) for i in range(len(tstemp))]
                normalize_ts_dict[obj] = temp                
        return normalize_ts_dict
    
    ###################################################################################################################################################
    # Input/Output
    ###################################################################################################################################################     
    def prepare_input(self,norml_ts=None,expname='ncrf36',orig=True,leftstart=None,pushfront=2):
        if expname=='ctl':
            if orig is True:
                dt = np.asarray(norml_ts['heatsum'][0])
                dtth = np.concatenate((norml_ts['theta'][0],norml_ts['heatsum'][0]),axis=1)
                dtthuvw = np.concatenate((norml_ts['u'][0],norml_ts['v'][0],norml_ts['w'][0],norml_ts['theta'][0],norml_ts['heatsum'][0]),axis=1)
                dtthuv = np.concatenate((norml_ts['u'][0],norml_ts['v'][0],norml_ts['theta'][0],norml_ts['heatsum'][0]),axis=1)
                uv = np.concatenate((norml_ts['u'][0],norml_ts['v'][0]),axis=1)
                uvw = np.concatenate((norml_ts['u'][0],norml_ts['v'][0],norml_ts['w'][0]),axis=1)
                dtuv = np.concatenate((norml_ts['u'][0],norml_ts['v'][0],norml_ts['heatsum'][0]),axis=1)
                dtuvw = np.concatenate((norml_ts['u'][0],norml_ts['v'][0],norml_ts['w'][0],norml_ts['heatsum'][0]),axis=1)
                dtthuvwqv = np.concatenate((norml_ts['u'][0],norml_ts['v'][0],norml_ts['w'][0],norml_ts['qv'][0],norml_ts['theta'][0],norml_ts['heatsum'][0]),axis=1)
                output_dict={'dt':dt,'dtth':dtth,'dtthuvw':dtthuvw,'dtthuv':dtthuv,'uv':uv,'uvw':uvw,'dtuv':dtuv,'dtuvw':dtuvw,'dtthuvwqv':dtthuvwqv}
                return output_dict
        else:
            if expname=='ncrf_36h':
                rsindx,expindx = 36,1
            elif expname=='ncrf_60h':
                rsindx,expindx = 60,2
            elif expname=='lwcrf':
                rsindx,expindx = 36,3
                        
            if orig is True:
                dt = np.asarray(norml_ts['heatsum'][expindx][rsindx-leftstart-pushfront:])
                dtth = np.concatenate((norml_ts['theta'][expindx][rsindx-leftstart-pushfront:],norml_ts['heatsum'][expindx][rsindx-leftstart-pushfront:]),axis=1)
                dtthuvw = np.concatenate((norml_ts['u'][expindx][rsindx-leftstart-pushfront:],norml_ts['v'][expindx][rsindx-leftstart-pushfront:],\
                                          norml_ts['w'][expindx][rsindx-leftstart-pushfront:],norml_ts['theta'][expindx][rsindx-leftstart-pushfront:],\
                                          norml_ts['heatsum'][expindx][rsindx-leftstart-pushfront:]),axis=1)
                dtthuv = np.concatenate((norml_ts['u'][expindx][rsindx-leftstart-pushfront:],norml_ts['v'][expindx][rsindx-leftstart-pushfront:],\
                                         norml_ts['theta'][expindx][rsindx-leftstart-pushfront:],norml_ts['heatsum'][expindx][rsindx-leftstart-pushfront:]),axis=1)
                uv = np.concatenate((norml_ts['u'][expindx][rsindx-leftstart-pushfront:],norml_ts['v'][expindx][rsindx-leftstart-pushfront:]),axis=1)
                uvw = np.concatenate((norml_ts['u'][expindx][rsindx-leftstart-pushfront:],norml_ts['v'][expindx][rsindx-leftstart-pushfront:],\
                                      norml_ts['w'][expindx][rsindx-leftstart-pushfront:]),axis=1)
                dtuv = np.concatenate((norml_ts['u'][expindx][rsindx-leftstart-pushfront:],norml_ts['v'][expindx][rsindx-leftstart-pushfront:],\
                                       norml_ts['heatsum'][expindx][rsindx-leftstart-pushfront:]),axis=1)
                dtuvw = np.concatenate((norml_ts['u'][expindx][rsindx-leftstart-pushfront:],norml_ts['v'][expindx][rsindx-leftstart-pushfront:],\
                                        norml_ts['w'][expindx][rsindx-leftstart-pushfront:],norml_ts['heatsum'][expindx][rsindx-leftstart-pushfront:]),axis=1)
                dtthuvwqv = np.concatenate((norml_ts['u'][expindx][rsindx-leftstart-pushfront:],norml_ts['v'][expindx][rsindx-leftstart-pushfront:],\
                                            norml_ts['w'][expindx][rsindx-leftstart-pushfront:],norml_ts['qv'][expindx][rsindx-leftstart-pushfront:],\
                                            norml_ts['theta'][expindx][rsindx-leftstart-pushfront:],norml_ts['heatsum'][expindx][rsindx-leftstart-pushfront:]),axis=1)
                output_dict={'dt':dt,'dtth':dtth,'dtthuvw':dtthuvw,'dtthuv':dtthuv,'uv':uv,'uvw':uvw,'dtuv':dtuv,'dtuvw':dtuvw,'dtthuvwqv':dtthuvwqv}
                return output_dict
            
    def create_input(self,PCAtimeseries=None,expname=None,leftstart=24,pushfront=2):
        inputpreproc = []
        for indx,expnameINT in enumerate(expname):
            inputpreproc.append(self.prepare_input(PCAtimeseries,expnameINT,True,leftstart,pushfront))
        mlr_inputtype = ['dt','dtth','dtthuvw','dtthuv','uv','uvw','dtuv','dtuvw','dtthuvwqv']
        mlr_inputdict = {}
        for TYPE in mlr_inputtype:
            print(TYPE)
            result = np.concatenate([exp[TYPE] for exp in inputpreproc],axis=0)
            mlr_inputdict[TYPE] = result
        return mlr_inputdict
    
    def produce_output_LT(self,norml_ts=None,expname=None,leadtime=None,nocomp=None,leftstart=24,pushfront=2,withtheta=True):
        def output_timediff(LT=None,inputdict=None,exp=None,leftstart=24,pushfront=2):
            if exp=='ctl':
                if withtheta is True:
                    a = np.concatenate([forward_diff(inputdict['u'][0],60*60,0,LT),forward_diff(inputdict['v'][0],60*60,0,LT),\
                                        forward_diff(inputdict['w'][0],60*60,0,LT),forward_diff(inputdict['theta'][0],60*60,0,LT)],axis=1)
                    azero = np.zeros((LT,nocomp[0]+nocomp[1]+nocomp[2]+nocomp[4]))
                else:
                    a = np.concatenate([forward_diff(inputdict['u'][0],60*60,0,LT),forward_diff(inputdict['v'][0],60*60,0,LT),forward_diff(inputdict['w'][0],60*60,0,LT)],axis=1)
                    azero = np.zeros((LT,nocomp[0]+nocomp[1]+nocomp[2]))
                return np.concatenate((a,azero),axis=0)
            else:
                if exp=='ncrf_36h':
                    rsindx,expindx = 36,1
                elif exp=='ncrf_60h':
                    rsindx,expindx = 60,2
                elif exp=='lwcrf':
                    rsindx,expindx = 36,3
                if withtheta is True:
                    a = np.concatenate([forward_diff(inputdict['u'][expindx][rsindx-leftstart-pushfront:],60*60,0,LT),forward_diff(inputdict['v'][expindx][rsindx-leftstart-pushfront:],60*60,0,LT),\
                                        forward_diff(inputdict['w'][expindx][rsindx-leftstart-pushfront:],60*60,0,LT),\
                                        forward_diff(inputdict['theta'][expindx][rsindx-leftstart-pushfront:],60*60,0,LT)],axis=1)
                    azero = np.zeros((LT,nocomp[0]+nocomp[1]+nocomp[2]+nocomp[4]))
                else:
                    a = np.concatenate([forward_diff(inputdict['u'][expindx][rsindx-leftstart-pushfront:],60*60,0,LT),forward_diff(inputdict['v'][expindx][rsindx-leftstart-pushfront:],60*60,0,LT),\
                                        forward_diff(inputdict['w'][expindx][rsindx-leftstart-pushfront:],60*60,0,LT)],axis=1)
                    azero = np.zeros((LT,nocomp[0]+nocomp[1]+nocomp[2]))
                return np.concatenate((a,azero),axis=0)
        result = []
        for index,expNAME in enumerate(expname):
            result.append(output_timediff(LT=leadtime,inputdict=norml_ts,exp=expNAME,leftstart=leftstart,pushfront=pushfront))
        del index,expNAME
        result_con = np.concatenate((result),axis=0)
        return result_con

class datacheck:
    def __init__(self,folderpath=None,pcapath=None,smooth24=False):
        self.folderpath=folderpath
        self.pcadict = read_and_proc.depickle(self.folderpath+pcapath)
        if smooth24:
            self.ctlflatvar = [read_and_proc.depickle(self.folderpath+str(['ctl'][i])+'_'+'preproc_dict1') for i in range(len(['ctl']))]
        else:
            self.ctlflatvar = [read_and_proc.depickle(self.folderpath+'uvwheat/'+str(['ctl'][i])+'_'+'preproc_dict1') for i in range(len(['ctl']))]
    
    def dudvdwVAR(self,dudvdwpath=None,vartest='w'):
        dudvdw = read_and_proc.depickle(self.folderpath+dudvdwpath)
        TESTu = [np.dot(forward_diff(self.pcadict[vartest].transform(self.ctlflatvar[0][vartest])[:,0:int(i)],60*60,0,1),(self.pcadict[vartest].components_[0:int(i)])) for i in np.linspace(0,60,31)]
        TESTu_var = [np.var(obj)/np.var(dudvdw['d'+str(vartest)]) for obj in TESTu]
        del TESTu
        gc.collect()
        return TESTu_var
    
    def dthdQVAR(self,dudvdwpath=None,vartest='th',smooth24=False):
        dudvdw = read_and_proc.depickle(self.folderpath+dudvdwpath)
        if vartest=='th':
            vartestPC='theta'
            if smooth24:
                TESTu = [np.dot(forward_diff(self.pcadict[vartestPC].transform(self.ctlflatvar[0][vartestPC])[:,0:int(i)],60*60,0,1),(self.pcadict[vartestPC].components_[0:int(i)])) for i in np.linspace(0,60,31)]
            else:
                folderpath='/work/FAC/FGSE/IDYST/tbeucler/default/freddy0218/TCGphy/2020_TC_CRF/dev/freddy0218/pca/output/uvwheat/'
                PCAtheta = read_and_proc.depickle(folderpath+'PCA/theta_PCA_dict1')
                folderpath='/work/FAC/FGSE/IDYST/tbeucler/default/freddy0218/TCGphy/2020_TC_CRF/dev/freddy0218/pca/output/flatvar/'
                thetavar = read_and_proc.depickle(folderpath+'theta_'+'preproc_dict1')
                TESTu = [np.dot(forward_diff(PCAtheta[vartestPC].transform(thetavar['ctlTHETA'])[:,0:int(i)],60*60,0,1),(PCAtheta[vartestPC].components_[0:int(i)])) for i in np.linspace(0,60,31)]
        elif vartest=='Q':
            vartestPC = 'heatsum'
            TESTu = [np.dot(forward_diff(self.pcadict[vartestPC].transform(self.ctlflatvar[0][vartestPC])[:,0:int(i)],60*60,0,1),\
                            (self.pcadict[vartestPC].components_[0:int(i)])) for i in np.linspace(0,60,31)]
        TESTu_var = [np.var(obj)/np.var(dudvdw['d'+str(vartest)]) for obj in TESTu]
        del TESTu
        gc.collect()
        return TESTu_var

