"""──────────────────────────────────────────────────────────────────────────┐
│ Loading necessary libraries to build and train model                       │
└──────────────────────────────────────────────────────────────────────────"""
import os,sys,gc,glob
import numpy as np
import pickle
import torch
from tqdm.auto import tqdm

sys.path.insert(1, '/work/FAC/FGSE/IDYST/tbeucler/default/freddy0218/TCGphy/2020_TC_CRF/dev/freddy0218/scikit/')
from tools import read_and_proc
#from tools.mlr import mlr
#from tools.preprocess import do_eof,preproc_maria,preproc_haiyan
sys.path.insert(2,'../../')
import read_stuff as read
import ts_models

"""──────────────────────────────────────────────────────────────────────────┐
│ Load settings                       │
└──────────────────────────────────────────────────────────────────────────"""
path = '/work/FAC/FGSE/IDYST/tbeucler/default/freddy0218/'
suffix = '_smooth_preproc_dict1b_g'
#enter = '/work/FAC/FGSE/IDYST/tbeucler/default/freddy0218/TCGphy/2020_TC_CRF/dev/freddy0218/pca/output/uvwheat/preproc2/'
#maria_u = [read_and_proc.depickle(path+'TCGphy/2020_TC_CRF/dev/freddy0218/pca/output/uvwheat/preproc2/'+str(lime)+suffix)['u'] for lime in tqdm(['ctl','ncrf_36h','ncrf_60h','ncrf_96h','lwcrf'])]
#divider = np.asarray([maria_u[0][12:].shape[0],maria_u[1][25:].shape[0],maria_u[2][49:].shape[0],maria_u[3][85:].shape[0],maria_u[4][25:].shape[0]]).cumsum()

expname=(str(sys.argv[1]))
splitnum=int(str(sys.argv[2]))
timelag=int(str(sys.argv[3]))
droprate=float(str(sys.argv[4]))

"""──────────────────────────────────────────────────────────────────────────┐
│ Loading input and outpus                    
└──────────────────────────────────────────────────────────────────────────"""
folder = '/work/FAC/FGSE/IDYST/tbeucler/default/freddy0218/TCGphy/2020_TC_CRF/dev/freddy0218/testML/output/maria/processed/timeseries/'#intermediate/'
Xtrain,Xvalid,Xtest,ytrain,yvalid,ytest = [],[],[],[],[],[]
for expname in [2,3,4]:
    obj = [read_and_proc.depickle(objs) for objs in sorted(glob.glob(folder+'inputoutput2/*val'+str(expname)))]
    Xtest.append(obj[0])
    Xtrain.append(obj[1])
    Xvalid.append(obj[2])
    ytest.append(obj[3])
    ytrain.append(obj[4])
    yvalid.append(obj[5])

TYPE = '3D'
if TYPE=='3D':
    folderpath='/work/FAC/FGSE/IDYST/tbeucler/default/freddy0218/TCGphy/2020_TC_CRF/dev/freddy0218/testML/output/maria/processed/'
    pcastore = read_and_proc.depickle(folderpath+'PCA/PCAdict3D')
elif TYPE=='2D':
    folderpath='/work/FAC/FGSE/IDYST/tbeucler/default/freddy0218/TCGphy/2020_TC_CRF/dev/freddy0218/testML/output/maria/processed/intermediate/'
    pcastore = read_and_proc.depickle(folderpath+'PCA/PCAdict')
    
folderpath=str('/work/FAC/FGSE/IDYST/tbeucler/default/freddy0218/TCGphy/2020_TC_CRF/dev/freddy0218/TCG_Rad_keras/store/')
folderpath2=str('/work/FAC/FGSE/IDYST/tbeucler/default/freddy0218/TCGphy/2020_TC_CRF/dev/freddy0218/testML/output/haiyan/processed/new3D/')
wcomps = [50,38,8]
upcs,vpcs,thpcs = pcastore['u'].components_[:wcomps[0]],pcastore['v'].components_[:wcomps[1]],pcastore['theta'].components_[:wcomps[2]]

X_totrain,y_totrain = read.train_optimizedMLR(folderpath,folderpath2).delete_padding(Xtrain[splitnum],ytrain[splitnum])#yall_orig[splitnum][23][0])
X_tovalid,y_tovalid = read.train_optimizedMLR(folderpath,folderpath2).delete_padding(Xvalid[splitnum],yvalid[splitnum])#yall_orig[splitnum][23][1])
X_totest,y_totest = read.train_optimizedMLR(folderpath,folderpath2).delete_padding(Xtest[splitnum],ytest[splitnum])#yall_orig[splitnum][23][2])
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

batch_size = 9
num_workers = 2
train_loader = torch.utils.data.DataLoader(
    dataset=train_data,
    batch_size=batch_size,
    shuffle=True)
val_loader = torch.utils.data.DataLoader(
    dataset=val_data,
    batch_size=batch_size,
    shuffle=False)
test_loader = torch.utils.data.DataLoader(
    dataset=test_data,
    batch_size=batch_size,
    shuffle=False)

import optuna

def objective(trial):
    models,losses = [],[]
    #droprate = trial.suggest_float("droprate",0.05,0.45)
    model = ts_models.OptimMLR_lwsw_3D_ts_dropout(droprate)
    lr = trial.suggest_float("lr",1e-6,1e-3)#,log=True)
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    criterion = torch.nn.L1Loss()
    n_epochs = 1500
    #lossfuncs = [torch.nn.L1Loss()]
    scheduler2 = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-8, max_lr=1e-4,cycle_momentum=False)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',min_lr=1e-12)
    #early_stopper = linear_models.EarlyStopping(patience=150, verbose=False, delta=1e-10, path='checkpoint.pt', trace_func=print)

    l2_lambda = trial.suggest_float("l2_lambda",0.01,0.02)
    #model,loss = train_model(model=model,train_data=data_loaders['train'],val_data=data_loaders['val'],optimizer=optimizer,scheduler=[scheduler,scheduler2],numepochs=num_epochs,early_stopper=None,variance_store=None,\
    #                         lossfunc=lossfuncs[0],regularization='L2',l1_lambda=0.1,l2_lambda=l2_lambda,trial=trial)
    #torch.save(model,'../tmp/bayesian/saved_model.8.'+str(trial.number)+'.pt')
    # Define Loss, Optimizer
    train_losses = []
    val_losses = []
    for epoch in range(1,n_epochs+1):
        loss = 0
        for features, labels in train_loader:
            optimizer.zero_grad() # Clears existing gradients from previous epoch
            output = model(features)
            batch_loss = criterion(output, labels.unsqueeze(1))
            batch_loss.backward()
            optimizer.step()
            loss += batch_loss.item()
        loss = loss/len(train_loader)
        train_losses.append(loss)
        val_loss = ts_models.eval_model(model,
                              val_loader,
                              criterion,
                             l2_lambda)
        val_losses.append(val_loss)
        if epoch%100 == 0:
            print('Epoch: {}/{}.............'.format(epoch, n_epochs))
            print("Loss: {:.4f}".format(loss))
        #if val_loss <= min(val_losses):
        #    torch.save(model,'best_model'+str(trial.number))
    #torch.save(model,'./tmp/bayesian/best_model.8.'+str(trial.number)+'.pt')
    return loss

study = optuna.create_study(directions=["minimize"])
study.optimize(objective, n_trials=10)#, timeout=300)
read_and_proc.save_to_pickle('../models/1d/mcdrop2/'+str(droprate)+'/'+str(splitnum)+'/bestparams.pkt',study,'PICKLE')
#study = read_and_proc.depickle('../tmp/mcdrop/'+str(expname)+'/0/'+str(droprate)+'/'+str(splitnum)+'/bestparams.pkt')

times = ['exp1a','exp1b','exp1c','exp1d','exp1e','exp1f','exp1g','exp1h','exp1i']
#times = ['exp1d','exp1e','exp1f','exp1g','exp1h','exp1i']#,'exp1d','exp1e']
for i in range(len(times)):
    models,losses = [],[]
    if expname=='lwswv':
        model = ts_models.OptimMLR_lwswv_3D_ts()
    elif expname=='lwswu':
        model = ts_models.OptimMLR_lwswu_3D_ts()
    elif expname=='lwsww':
        model = ts_models.OptimMLR_lwsww_3D_ts()
    elif expname=='lwswth':
        model = ts_models.OptimMLR_lwswth_3D_ts()
    elif expname=='lwswhdia':
        model = ts_models.OptimMLR_lwswhdia_3D_ts()
    elif expname=='lwsw_drop':
        model = ts_models.OptimMLR_lwsw_3D_ts_dropout(droprate)
    model = ts_models.OptimMLR_lwsw_3D_ts_dropout(droprate)
    optimizers = [torch.optim.Adam(model.parameters(), lr=study.best_params['lr'])]#, optim.AdaBound(model.parameters(),lr=1e-7)] 1e-6 [torch.optim.Adam(model.parameters(),lr=0.5e-5),torch.optim.SGD(model.parameters(),lr=0.5e-5,momentum=0.8)]
    loss = torch.nn.MSELoss()
    for optimizer in optimizers:
        scheduler2 = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-8, max_lr=5e-5,cycle_momentum=False) #1e-9/1e-5
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',min_lr=1e-12)  #1e-18
        num_epochs = 1000*20#26
        early_stopper = ts_models.EarlyStopping(patience=250, verbose=False, delta=1e-7, path='checkpoint.pt', trace_func=print)#EarlyStopper(patience=8, min_delta=1e-3)
        #variance_store = [varu,varv,varw,varth]
        #variance_store = [varu,varv,varth]
        model,loss = ts_models.train_model(model=model,optimizer=optimizer,scheduler=[scheduler,scheduler2],numepochs=num_epochs,early_stopper=early_stopper,variance_store=None,\
                                 lossfunc=loss,train_loader=train_loader,val_loader=val_loader,test_loader=test_loader,l2_lambda=study.best_params['l2_lambda'])
        models.append(model)
        losses.append(loss)
    #torch.save(models, '../tmp/torch_try/ts/'+str(expname)+'/0/'+'models'+str(splitnum)+'_'+str(expname)+'3dnonln_1115_'+str(times[i])+'.pt')
    #read_and_proc.save_to_pickle('../tmp/torch_try/ts/'+str(expname)+'/0/'+'losses'+str(splitnum)+'_'+str(expname)+'3dnonln_1115_'+str(times[i])+'.pkt',losses,'PICKLE')
    
    torch.save(models, '../models/1d/mcdrop2/'+str(droprate)+'/'+str(splitnum)+'/modelstest'+str(splitnum)+'_lwswts_1115_'+str(times[i])+'.pt')
    read_and_proc.save_to_pickle('../models/1d/mcdrop2/'+str(droprate)+'/'+str(splitnum)+'/lossestest'+str(splitnum)+'_lwswts_1115_'+str(times[i])+'.pkt',losses,'PICKLE')
