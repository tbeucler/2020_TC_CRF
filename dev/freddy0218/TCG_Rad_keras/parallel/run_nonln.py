"""──────────────────────────────────────────────────────────────────────────┐
│ Loading necessary libraries to build and train model                       │
└──────────────────────────────────────────────────────────────────────────"""
import os,sys,gc
import numpy as np
import pickle
import torch
from tqdm.auto import tqdm

sys.path.insert(1, '/work/FAC/FGSE/IDYST/tbeucler/default/freddy0218/TCGphy/2020_TC_CRF/dev/freddy0218/scikit/')
from tools import derive_var,read_and_proc
from tools.mlr import mlr
from tools.preprocess import do_eof,preproc_maria,preproc_haiyan
sys.path.insert(2,'../')
import read_stuff as read
import nonlinear_models
from livelossplot import PlotLosses

"""──────────────────────────────────────────────────────────────────────────┐
│ Load settings                       │
└──────────────────────────────────────────────────────────────────────────"""
path = '/work/FAC/FGSE/IDYST/tbeucler/default/freddy0218/'
suffix = '_smooth_preproc_dict1b_g'
#a = [read_and_proc.depickle(path+'TCGphy/2020_TC_CRF/dev/freddy0218/testML/output/haiyan/processed/uvwheat/'+'mem'+str(lime)+suffix)['u'].shape for lime in tqdm(range(1,21))]
# divide experiments reference
#divider = np.asarray([aobj[0] for aobj in a]).cumsum()

nonln_num=int(str(sys.argv[1]))
timelag=int(str(sys.argv[2]))
"""──────────────────────────────────────────────────────────────────────────┐
│ Loading input and outpus                    
└──────────────────────────────────────────────────────────────────────────"""
folderpath='/work/FAC/FGSE/IDYST/tbeucler/default/freddy0218/TCGphy/2020_TC_CRF/dev/freddy0218/TCG_Rad_keras/'
folderpath2='/work/FAC/FGSE/IDYST/tbeucler/default/freddy0218/TCGphy/2020_TC_CRF/dev/freddy0218/testML/output/haiyan/processed/new3D/'
Xtrain,Xvalid,Xtest,yall = read.train_optimizedMLR(folderpath,folderpath2,'rhorig','rhorig','3D').read_Xy(num=33,needorig='No')
pcastore = read.train_optimizedMLR(folderpath,folderpath2,'rhorig','rhorig','3D').pcastore
wcomps = [50,38,8]
upcs,vpcs,thpcs = pcastore['u'].components_[:wcomps[0]],pcastore['v'].components_[:wcomps[1]],pcastore['theta'].components_[:wcomps[2]]

for splitnum in range(17,33):
    X_totrain,y_totrain = read.train_optimizedMLR(folderpath,folderpath2).delete_padding(Xtrain[splitnum]['lwswdtthuvw'],yall[splitnum][timelag][0])#yall_orig[splitnum][23][0])
    X_tovalid,y_tovalid = read.train_optimizedMLR(folderpath,folderpath2).delete_padding(Xvalid[splitnum]['lwswdtthuvw'],yall[splitnum][timelag][1])#yall_orig[splitnum][23][1])
    X_totest,y_totest = read.train_optimizedMLR(folderpath,folderpath2).delete_padding(Xtest[splitnum]['lwswdtthuvw'],yall[splitnum][timelag][2])#yall_orig[splitnum][23][2])
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
    ###################################################################################
    # Eigenvectors for Custom Loss in Physical Space
    ###################################################################################
    #eigenvectors = [torch.FloatTensor(obj).to(calc_device) for obj in [upcs,vpcs,thpcs]]#[upcs,vpcs,wpcs,thpcs]]
    ###################################################################################
    # Variances to calculate r2 in Physical Space
    ###################################################################################
    #varu,varv,varw,varth = np.var(yTRUTH['train'][splitnum]['du']),np.var(yTRUTH['train'][splitnum]['dv']),np.var(yTRUTH['train'][splitnum]['dw']),np.var(yTRUTH['train'][splitnum]['dth'])
    #varu,varv,varth = np.var(yTRUTH['train'][splitnum]['du']),np.var(yTRUTH['train'][splitnum]['dv']),np.var(yTRUTH['train'][splitnum]['dth'])
    batch_size = 9
    num_workers = 2
    train_loader = torch.utils.data.DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_data,
        batch_size=batch_size,
        shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        shuffle=False
    )
    
    times = ['exp1a','exp1b','exp1c']#,'exp1d','exp1e']
    #times = ['exp2a','exp2b','exp2c']#,'exp1d','exp1e']
    for i in range(len(times)):
        models,losses = [],[]
        #model = OptimMLR_all_2D()
        model = nonlinear_models.OptimMLR_all_3D_lwswv(int(nonln_num))
        optimizers = [torch.optim.Adam(model.parameters(), lr=1e-7)]#, optim.AdaBound(model.parameters(),lr=1e-7)] 1e-6
        loss = torch.nn.MSELoss()
        for optimizer in optimizers:
            scheduler2 = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-9, max_lr=1e-5,cycle_momentum=False)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',min_lr=1e-18)
            num_epochs = 1000*10#26
            early_stopper = nonlinear_models.EarlyStopping(patience=50, verbose=False, delta=1e-7, path='checkpoint.pt', trace_func=print)#EarlyStopper(patience=8, min_delta=1e-3)
            #variance_store = [varu,varv,varw,varth]
            #variance_store = [varu,varv,varth]
            model,loss = nonlinear_models.train_model(model=model,optimizer=optimizer,scheduler=[scheduler,scheduler2],numepochs=num_epochs,early_stopper=early_stopper,variance_store=None,\
                                     lossfunc=loss,train_loader=train_loader,val_loader=val_loader,test_loader=test_loader)
            models.append(model)
            losses.append(loss)
        torch.save(models, '../tmp/torch_try/lwswv/'+str(nonln_num)+'/'+'models'+str(splitnum)+'_lwsw3dnonln_1115_'+str(times[i])+'.pt')
        read_and_proc.save_to_pickle('../tmp/torch_try/lwswv/'+str(nonln_num)+'/'+'losses'+str(splitnum)+'_lwsw3dnonln_1115_'+str(times[i])+'.pkt',losses,'PICKLE')
