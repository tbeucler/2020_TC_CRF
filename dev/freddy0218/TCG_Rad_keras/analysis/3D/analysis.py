import os,sys,gc
import numpy as np
import pickle
import torch
from tqdm.auto import tqdm

sys.path.insert(1, '/work/FAC/FGSE/IDYST/tbeucler/default/freddy0218/TCGphy/2020_TC_CRF/dev/freddy0218/scikit/')
from tools import derive_var,read_and_proc
from tools.mlr import mlr
from tools.preprocess import do_eof,preproc_maria,preproc_haiyan
sys.path.insert(2, '../')
import read_stuff as read
import glob

def output_processed(Xtrain=None,Xvalid=None,Xtest=None,yall=None,folderpath=None,folderpath2=None,splitnum=None):
    """──────────────────────────────────────────────────────────────────────────┐
    │ Which split will we be testing                     
    └──────────────────────────────────────────────────────────────────────────"""
    splitnum = splitnum
    """──────────────────────────────────────────────────────────────────────────┐
    │ Remove zero values                    
    └──────────────────────────────────────────────────────────────────────────"""
    X_totrain,y_totrain = read.train_optimizedMLR(folderpath,folderpath2,'rh').delete_padding(Xtrain[splitnum]['lwswdtthuvwqv'],yall[splitnum][23][0])
    X_tovalid,y_tovalid = read.train_optimizedMLR(folderpath,folderpath2,'rh').delete_padding(Xvalid[splitnum]['lwswdtthuvwqv'],yall[splitnum][23][1])
    X_totest,y_totest = read.train_optimizedMLR(folderpath,folderpath2,'rh').delete_padding(Xtest[splitnum]['lwswdtthuvwqv'],yall[splitnum][23][2])
    
    calc_device = 'cpu'
    ###################################################################################
    # Convert numpy arrays into tensors
    ###################################################################################
    train_Xtensor = torch.FloatTensor(X_totrain).to(calc_device)
    train_ytensor = torch.FloatTensor(y_totrain).to(calc_device)
    val_Xtensor = torch.FloatTensor(X_tovalid).to(calc_device)
    val_ytensor = torch.FloatTensor(y_tovalid).to(calc_device)
    test_Xtensor = torch.FloatTensor(X_totest).to(calc_device)
    test_ytensor = torch.FloatTensor(y_totest).to(calc_device)
    train_data = torch.utils.data.TensorDataset(train_Xtensor, train_ytensor)
    val_data = torch.utils.data.TensorDataset(val_Xtensor, val_ytensor)
    test_data = torch.utils.data.TensorDataset(test_Xtensor, test_ytensor)
    
    modelpaths = sorted(glob.glob('../tmp/torch_try/1108/'+str(splitnum)+'/models*'))
    model = [torch.load(obj) for obj in modelpaths]
    
    return {'train':train_Xtensor,'val':val_Xtensor,'test':test_Xtensor},{'train':train_ytensor,'val':val_ytensor,'test':test_ytensor},{'train':train_data,'val':val_data,'test':test_data},model

def _get_exp_name(splitnum=None,folder=2,folderpath=None):
    return sorted(glob.glob(folderpath+'pca/X/random/'+str(folder)+'/Xtrain*'))[splitnum][:-7].split('/')[-1][6:],sorted(glob.glob(folderpath+'pca/X/random/'+str(folder)+'/Xtrain*'))[splitnum][:-7].split('/')[-1][6:].split('_')

def real_random(index=None,folder=2,folderpath=None):
    toextract = _get_exp_name(index,folder,folderpath)[0]
    # X
    Xtestpath,Xtrainpath,Xvalidpath = sorted(glob.glob(folderpath+'pca/X/random/'+str(folder)+'/*'+str(toextract)+'*'))
    Xtest,Xtrain,Xvalid = [read_and_proc.depickle(obj) for obj in[Xtestpath,Xtrainpath,Xvalidpath]]
    # y
    yallpath = sorted(glob.glob(folderpath+'pca/y/random/'+str(folder)+'/*'+str(toextract)+'*'))
    yall = read_and_proc.depickle(yallpath[0])
    return Xtest,Xtrain,Xvalid,yall

def store_r2(pcastore=None,yTRUTH=None,model=None,splitnum=30,Xtensor=None):
    folderpath='/work/FAC/FGSE/IDYST/tbeucler/default/freddy0218/TCGphy/2020_TC_CRF/dev/freddy0218/TCG_Rad_keras/'
    folderpath2='/work/FAC/FGSE/IDYST/tbeucler/default/freddy0218/TCGphy/2020_TC_CRF/dev/freddy0218/testML/output/haiyan/processed/intermediate/'
    #Xtensor,ytensor,data,models = output_processed(Xtrain,Xvalid,Xtest,yall,folderpath,folderpath2,splitnum)
    orig_th = np.dot(haiyanmodelsDICT['model'][splitnum][0].predict(orig_testY[splitnum][1])[:,26+18+48:26+18+48+14],pcastore['theta'].components_[:14])
    orig_w = np.dot(haiyanmodelsDICT['model'][splitnum][0].predict(orig_testY[splitnum][1])[:,26+18:26+18+48],pcastore['w'].components_[:48])
    orig_v = np.dot(haiyanmodelsDICT['model'][splitnum][0].predict(orig_testY[splitnum][1])[:,26:26+18],pcastore['v'].components_[:18])
    orig_u = np.dot(haiyanmodelsDICT['model'][splitnum][0].predict(orig_testY[splitnum][1])[:,:26],pcastore['u'].components_[:26])
    origs = [orig_u,orig_v,orig_w,orig_th]
    del orig_u, orig_v, orig_w, orig_th
    gc.collect()

    bbbbbb_th = np.dot(model[0](Xtensor['test']).detach().numpy()[:,26+18+48:26+18+48+14],pcastore['theta'].components_[:14])
    bbbbbb_w = np.dot(model[0](Xtensor['test']).detach().numpy()[:,26+18:26+18+48],pcastore['w'].components_[:48])
    bbbbbb_v = np.dot(model[0](Xtensor['test']).detach().numpy()[:,26:26+18],pcastore['v'].components_[:18])
    bbbbbb_u = np.dot(model[0](Xtensor['test']).detach().numpy()[:,:26],pcastore['u'].components_[:26])
    
    uper = [r2_score(yTRUTH['test'][splitnum]['du'],origs[0]),r2_score(yTRUTH['test'][splitnum]['du'],bbbbbb_u)]
    vper = [r2_score(yTRUTH['test'][splitnum]['dv'],origs[1]),r2_score(yTRUTH['test'][splitnum]['dv'],bbbbbb_v)]
    wper = [r2_score(yTRUTH['test'][splitnum]['dw'],origs[2]),r2_score(yTRUTH['test'][splitnum]['dw'],bbbbbb_w)]
    thper = [r2_score(yTRUTH['test'][splitnum]['dth'],origs[3]),r2_score(yTRUTH['test'][splitnum]['dth'],bbbbbb_th)]
    return {'u':uper[0],'v':vper[0],'w':wper[0],'theta':thper[0]},{'u':uper[1],'v':vper[1],'w':wper[1],'theta':thper[1]}

