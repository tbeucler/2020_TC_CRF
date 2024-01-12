import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import os,sys,gc
import numpy as np
import pickle
import torch
from tqdm.auto import tqdm
import glob

sys.path.insert(1, '/work/FAC/FGSE/IDYST/tbeucler/default/freddy0218/TCGphy/2020_TC_CRF/dev/freddy0218/scikit/')
from tools import derive_var,read_and_proc
from tools.mlr import mlr
#from tools.preprocess import preproc_maria,preproc_haiyan
sys.path.insert(1, '../')
import read_stuff as read
import ts_models
import properscoring as ps

calc_device='cpu'
class OptimMLR_lwsw_3D_ts_dropout_small_dropemulate(torch.nn.Module):
    def __init__(self,droprate,brchindex):
        #super(OptimMLR_all_2D, self).__init__()
        super(OptimMLR_lwsw_3D_ts_dropout_small_dropemulate, self).__init__()
        
        self.brchindex = list(np.asarray(brchindex).cumsum())
        ############################################################
        # Input channels
        ############################################################
        brchsize = [brchindex[-2],brchindex[-1]]#[50,38,91,8,82,20,20]
        self.dense1 = torch.nn.Linear(brchsize[0], 1)
        self.dense2 = torch.nn.Linear(brchsize[1], 1)
        self.dropout1 = torch.nn.Dropout(droprate)
        self.dropout2 = torch.nn.Dropout(droprate)
        
    def forward(self,X):
        #brchindex = list(np.asarray([0,50,38,91,8,82,20,20]).cumsum())
        X_u, X_v, X_w, X_th = X[:,self.brchindex[0]:self.brchindex[1]],X[:,self.brchindex[1]:self.brchindex[2]],X[:,self.brchindex[2]:self.brchindex[3]],X[:,self.brchindex[3]:self.brchindex[4]]
        X_hdia, X_lw, X_sw = X[:,self.brchindex[4]:self.brchindex[5]],X[:,self.brchindex[5]:self.brchindex[6]],X[:,self.brchindex[6]:self.brchindex[7]]
        ############################################################
        # Optimal PC layer
        ############################################################
        X_lwc = self.dropout1(X_lw)
        #bestlw = self.dense1(X_lwc)
        X_swc = self.dropout2(X_sw)
        #bestsw = self.dense2(X_swc)
        return X_lwc,X_swc
    
class OptimMLR_lwsw_3D_ts_dropout_small(torch.nn.Module):
    def __init__(self,droprate,brchindex):
        #super(OptimMLR_all_2D, self).__init__()
        super(OptimMLR_lwsw_3D_ts_dropout_small, self).__init__()
        self.brchindex = list(np.asarray(brchindex).cumsum())
        ############################################################
        # Input channels
        ############################################################
        brchsize = [brchindex[-2],brchindex[-1]]#[50,38,91,8,82,20,20]
        self.dense1 = torch.nn.Linear(brchsize[0], 1)
        self.dense2 = torch.nn.Linear(brchsize[1], 1)
        self.dropout1 = torch.nn.Dropout(droprate)
        self.dropout2 = torch.nn.Dropout(droprate)

        
    def forward(self,X):
        #brchindex = list(np.asarray([0,50,38,91,8,82,20,20]).cumsum())
        X_u, X_v, X_w, X_th = X[:,self.brchindex[0]:self.brchindex[1]],X[:,self.brchindex[1]:self.brchindex[2]],X[:,self.brchindex[2]:self.brchindex[3]],X[:,self.brchindex[3]:self.brchindex[4]]
        X_hdia, X_lw, X_sw = X[:,self.brchindex[4]:self.brchindex[5]],X[:,self.brchindex[5]:self.brchindex[6]],X[:,self.brchindex[6]:self.brchindex[7]]
        ############################################################
        # Optimal PC layer
        ############################################################
        X_lwc = self.dropout1(X_lw)
        bestlw = self.dense1(X_lwc)
        X_swc = self.dropout2(X_sw)
        bestsw = self.dense2(X_swc)
        return bestlw,bestsw

class OptimMLR_lwsw_3D_ts_dropout_small2(torch.nn.Module):
    def __init__(self,droprate):
        #super(OptimMLR_all_2D, self).__init__()
        super(OptimMLR_lwsw_3D_ts_dropout_small2, self).__init__()
        self.dropout3 = torch.nn.Dropout(droprate)
        self.denseout = torch.nn.Linear(2,1)#106)
        
    def forward(self,X):
        bestPC = self.dropout3(X)
        outpred = self.denseout(bestPC)
        return outpred
    
def get_innertimeseries(filepath,splitnum=None,droprates=None,droprate_index=7,bestmodel=None,brchindex=[0,50,38,91,8,82,20,20],explag=23):
    modelsmall = OptimMLR_lwsw_3D_ts_dropout_small(droprates[droprate_index],brchindex)
    model_smalldict = modelsmall.state_dict()
    pretrained_dict = bestmodel.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_smalldict}
    model_smalldict.update(pretrained_dict)
    modelsmall.load_state_dict(model_smalldict)
    
    Xtrain = read_and_proc.depickle(glob.glob(filepath+'/X/'+str(splitnum)+'/*')[0])['uvwthhdialwsw']['train']
    Xvalid = read_and_proc.depickle(glob.glob(filepath+'/X/'+str(splitnum)+'/*')[0])['uvwthhdialwsw']['valid']
    Xtest = read_and_proc.depickle(glob.glob(filepath+'/X/'+str(splitnum)+'/*')[0])['uvwthhdialwsw']['test']
    ytrain = read_and_proc.depickle(glob.glob(filepath+'/y/'+str(splitnum)+'/*')[0])['train'][explag]
    yvalid = read_and_proc.depickle(glob.glob(filepath+'/y/'+str(splitnum)+'/*')[0])['valid'][explag]
    ytest = read_and_proc.depickle(glob.glob(filepath+'/y/'+str(splitnum)+'/*')[0])['test'][explag]
    
    X_totrain, y_totrain = read.delete_padding(Xtrain,ytrain)
    X_tovalid, y_tovalid = read.delete_padding(Xvalid,yvalid)
    X_totest, y_totest = read.delete_padding(Xtest,ytest)
    
    test_Xtensor = torch.FloatTensor(X_totest).to(calc_device)
    #test_ytensor = torch.FloatTensor(ytest).to(calc_device)
    lwdrop,swdrop = modelsmall.eval()(test_Xtensor)
    lwdrop,swdrop = lwdrop.detach().numpy(),swdrop.detach().numpy()
    return lwdrop,swdrop

def model_outweights(model=None):
    params,names = [],[]
    for name, param in model.named_parameters():
        if ".weight" not in name:
            continue            
        else:
            params.append(param)
            names.append(name)
    return params, names

def long_MariaExps(array=None,start=None):
    haiyan_temparray = [(array[0][12:,:])[15:,:],array[1][int(start[0]):,:],array[2][int(start[1]):,:],array[3][int(start[2]):,:],array[4][int(start[3]):,:]]
    #haiyan_temparray = [(array[0][12:,:]),array[1][int(start[0]):,:],array[2][int(start[1]):,:],array[3][int(start[2]):,:],array[4][int(start[3]):,:]]
    del array
    gc.collect()
    return haiyan_temparray#np.con

def find_start(haiyan_data=None,ref1=0,testexp=1):
    for i in range(len(haiyan_data[ref1])):
        if str((haiyan_data[ref1][i]==haiyan_data[testexp][i]).all())=='True':
            continue
        else:
            break
    return i