import os,sys,gc
import numpy as np
import pickle
import torch
import proplot as plot
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import pandas as pd
import seaborn as sns
sys.path.insert(1, '/work/FAC/FGSE/IDYST/tbeucler/default/freddy0218/TCGphy/2020_TC_CRF/dev/freddy0218/scikit/')
from tools import derive_var,read_and_proc
from tools.mlr import mlr
sys.path.insert(1, '../')
import read_stuff as read
sys.path.insert(1, '/work/FAC/FGSE/IDYST/tbeucler/default/freddy0218/TCGphy/2020_TC_CRF/dev/freddy0218/TCG_Rad_keras/parallel/')
import linear_models
import properscoring as ps
import glob
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def model_outweights(model=None):
    params,names = [],[]
    for name, param in model.named_parameters():
        if ".weight" not in name:
            continue
        else:
            params.append(param)
            names.append(name)
    return params, names

def grab_predictions(model=None,Xtensors=None,trial=20,TYPE='all'):
    if TYPE=='all':
        return [np.squeeze(model.train()(Xtensors).detach().numpy().transpose()) for i in range(trial)],np.nanmean(np.asarray([np.squeeze(model.train()(Xtensors).detach().numpy().transpose()) for i in range(trial)]),axis=0)
    elif TYPE=='Drop1_2':
        model.dropout1.train()
        model.dropout2.train()
        storeweights = [model_outweights(model) for i in range(trial)]
        return [np.squeeze(model(Xtensors).detach().numpy().transpose()) for i in range(trial)],np.nanmean(np.asarray([np.squeeze(model(Xtensors).detach().numpy().transpose()) for i in range(trial)]),axis=0),storeweights
    
def get_meanr2(X=None,y=None):
    return r2_score(y,X)
def get_meanrmse(X=None,y=None):
    return np.sqrt(mean_squared_error(y,X))
def get_meanmae(X=None,y=None):
    return mean_absolute_error(y,X)

def get_performances(folderpath=str('/work/FAC/FGSE/IDYST/tbeucler/default/freddy0218/TCGphy/2020_TC_CRF/dev/freddy0218/TCG_Rad_keras/store/'),
                     folderpath2=str('/work/FAC/FGSE/IDYST/tbeucler/default/freddy0218/TCGphy/2020_TC_CRF/dev/freddy0218/testML/output/haiyan/processed/new3D/'),
                     droprate=None,Xtrain=None,Xvalid=None,Xtest=None,yall=None,metric='r2',trailnums=50,withspread=False):
    trailnums = trailnums
    allstoredicts_drop1_2 = []
    for splitnum in tqdm(range(33)):
        X_totrain,y_totrain = read.train_optimizedMLR(folderpath,folderpath2).delete_padding(Xtrain[splitnum]['lwswdtthuvw'],yall[splitnum][23][0])#yall_orig[splitnum][23][0])
        X_tovalid,y_tovalid = read.train_optimizedMLR(folderpath,folderpath2).delete_padding(Xvalid[splitnum]['lwswdtthuvw'],yall[splitnum][23][1])#yall_orig[splitnum][23][1])
        X_totest,y_totest = read.train_optimizedMLR(folderpath,folderpath2).delete_padding(Xtest[splitnum]['lwswdtthuvw'],yall[splitnum][23][2])#yall_orig[splitnum][23][2])
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
        ###################################################################################
        # Models
        ###################################################################################
        models = [torch.load(obj) for obj in glob.glob('../tmp/torch_try/ts/lwsw_drop/0/'+str(droprate)+'/modelstest'+str(splitnum)+'_*')]
        ###################################################################################
        # Make predictions
        ###################################################################################
        alltrains,meantrains,allvals,meanvals,alltests,meantests,r2trains,r2vals,r2tests,weights = [],[],[],[],[],[],[],[],[],[]
        spreaddicts,truth = [],[]
        for model in models:
            alltrain,meantrain,weight_train = grab_predictions(model[0],train_Xtensor,trailnums,'Drop1_2')
            allval,meanval,weight_val = grab_predictions(model[0],val_Xtensor,trailnums,'Drop1_2')
            alltest,meantest,weight_test = grab_predictions(model[0],test_Xtensor,trailnums,'Drop1_2')
            if metric=='r2':
                meanr2_train = get_meanr2(meantrain,y_totrain)
                meanr2_val = get_meanr2(meanval,y_tovalid)
                meanr2_test = get_meanr2(meantest,y_totest)
            elif metric=='rmse':
                meanr2_train = get_meanrmse(meantrain,y_totrain)
                meanr2_val = get_meanrmse(meanval,y_tovalid)
                meanr2_test = get_meanrmse(meantest,y_totest)
            elif metric=='mae':
                meanr2_train = get_meanmae(meantrain,y_totrain)
                meanr2_val = get_meanmae(meanval,y_tovalid)
                meanr2_test = get_meanmae(meantest,y_totest)
            elif metric=='crps':
                meanr2_train = ps.crps_ensemble(np.asarray(y_totrain).transpose(),np.asarray(alltrain).transpose()).mean()
                meanr2_val = ps.crps_ensemble(np.asarray(y_tovalid).transpose(),np.asarray(allval).transpose()).mean()
                meanr2_test = ps.crps_ensemble(np.asarray(y_totest).transpose(),np.asarray(alltest).transpose()).mean()
            if withspread:
                meanstd_train = np.std(np.asarray(alltrain),axis=0)
                meanstd_val = np.std(np.asarray(allval),axis=0)
                meanstd_test = np.std(np.asarray(alltest),axis=0)
                meanstd_dict = {'train':meanstd_train,'valid':meanstd_val,'test':meanstd_test}
            else:
                continue
            alltrains.append(alltrain)
            meantrains.append(meantrain)
            allvals.append(allval)
            meanvals.append(meanval)
            alltests.append(alltest)
            meantests.append(meantest)
            r2trains.append(meanr2_train)
            r2vals.append(meanr2_val)
            r2tests.append(meanr2_test)
            weights.append({'train':weight_train,'val':weight_val,'test':weight_test})
            spreaddicts.append(meanstd_dict)
            truth.append({'train':y_totrain,'valid':y_tovalid,'test':y_totest})
        
        allstoredicts_drop1_2.append({'models':models,'alltrains':alltrains,'meantrains':meantrains,'allvals':allvals,'meanvals':meanvals,'alltests':alltests,'meantests':meantests,'r2trains':r2trains,'r2vals':r2vals,'r2tests':r2tests,'weights':weights,'spreads':spreaddicts,'truth':truth})
    return allstoredicts_drop1_2
