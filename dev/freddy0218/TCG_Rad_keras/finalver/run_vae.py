"""──────────────────────────────────────────────────────────────────────────┐
│ Loading necessary libraries to build and train model                       │
└──────────────────────────────────────────────────────────────────────────"""
import os,sys,gc
import numpy as np
import pickle
import torch
from tqdm.auto import tqdm

sys.path.insert(1, '/work/FAC/FGSE/IDYST/tbeucler/default/freddy0218/TCGphy/2020_TC_CRF/dev/freddy0218/scikit/')
from tools import read_and_proc
#from tools.mlr import mlr
#from tools.preprocess import do_eof,preproc_maria,preproc_haiyan
import read_stuff as read
import ts_models
import glob,os,vae

print(sys.argv)
expname=(str(sys.argv[1]))
splitnum=int(str(sys.argv[2]))
timelag=int(str(sys.argv[3]))
losscoeff =float(str(sys.argv[4]))

def prepare_tensors(filepath='./haiyan_store',splitnum=None,explag=23,notensor='No'):
    if './haiyan_store' in filepath:
        Xtrain = read_and_proc.depickle(glob.glob(filepath+'/X/'+str(splitnum)+'/*intensity*')[0])['uvwthhdialwsw']['train']
        Xvalid = read_and_proc.depickle(glob.glob(filepath+'/X/'+str(splitnum)+'/*intensity*')[0])['uvwthhdialwsw']['valid']
        Xtest = read_and_proc.depickle(glob.glob(filepath+'/X/'+str(splitnum)+'/*intensity*')[0])['uvwthhdialwsw']['test']
        ytrain = read_and_proc.depickle(glob.glob(filepath+'/y/'+str(splitnum)+'/*intensity*')[0])['train'][explag]
        yvalid = read_and_proc.depickle(glob.glob(filepath+'/y/'+str(splitnum)+'/*intensity*')[0])['valid'][explag]
        ytest = read_and_proc.depickle(glob.glob(filepath+'/y/'+str(splitnum)+'/*intensity*')[0])['test'][explag]
    elif './maria_store' in filepath:
        Xtrain = read_and_proc.depickle(glob.glob(filepath+'/X/'+str(splitnum)+'/*corr*')[0])['uvwthhdialwsw']['train']
        Xvalid = read_and_proc.depickle(glob.glob(filepath+'/X/'+str(splitnum)+'/*corr*')[0])['uvwthhdialwsw']['valid']
        Xtest = read_and_proc.depickle(glob.glob(filepath+'/X/'+str(splitnum)+'/*corr*')[0])['uvwthhdialwsw']['test']
        ytrain = read_and_proc.depickle(glob.glob(filepath+'/y/'+str(splitnum)+'/*corr*')[0])['train'][explag]
        yvalid = read_and_proc.depickle(glob.glob(filepath+'/y/'+str(splitnum)+'/*corr*')[0])['valid'][explag]
        ytest = read_and_proc.depickle(glob.glob(filepath+'/y/'+str(splitnum)+'/*corr*')[0])['test'][explag]
    
    X_totrain, y_totrain = read.delete_padding(Xtrain,ytrain)
    X_tovalid, y_tovalid = read.delete_padding(Xvalid,yvalid)
    X_totest, y_totest = read.delete_padding(Xtest,ytest)
    
    if notensor=='No':
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
        
        return train_data,val_data,test_data #{'train':[train_Xtensor,train_ytensor],'valid':[val_Xtensor,val_ytensor],'test':[test_Xtensor,test_ytensor]}
    elif notensor=='Yes':
        return {'train':[X_totrain, y_totrain],'valid':[X_tovalid, y_tovalid],'test':[X_totest, y_totest]}

filepath = './maria_store/'
train_data,val_data,test_data = prepare_tensors(filepath,splitnum,timelag,'No')

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
if filepath=='./maria_store/':
    nummem = [0,50,26,50,50,50,10,10]
elif filepath=='./haiyan_store/':
    #nummem = [0,50,38,91,8,82,20,20]
    nummem = [0,50,38,50,8,50,20,20]

def objective(trial):
    models,losses = [],[]
    model = vae.VAE(nummem[-2],nummem[-1],1,1,1,nummem)
    #droprate = trial.suggest_float("droprate",0.05,0.45)
    lr = trial.suggest_float("lr",1e-6,1e-3)#,log=True)
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    criterion = vae.vae_loss
    n_epochs = 500
    scheduler2 = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-8, max_lr=1e-4,cycle_momentum=False)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',min_lr=1e-12)

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
            reconX,mu1,logvar1,mu2,logvar2 = model(features)
            batch_loss,_,_ = vae.vae_loss(reconX, labels.unsqueeze(1),mu1,logvar1,mu2,logvar2,losscoeff)
            batch_loss.backward()
            optimizer.step()
            loss += batch_loss.item()
        loss = loss/len(train_loader)
        train_losses.append(loss)
        criterion = vae.vae_loss
        val_loss,_,_ = vae.eval_model(model,
                              val_loader,
                              criterion,
                             l2_lambda,
                                  losscoeff)
        val_losses.append(val_loss)
        if epoch%100 == 0:
            print('Epoch: {}/{}.............'.format(epoch, n_epochs))
            print("Loss: {:.4f}".format(loss))
        #if val_loss <= min(val_losses):
        #    torch.save(model,'best_model'+str(trial.number))
    #torch.save(model,'./tmp/bayesian/best_model.8.'+str(trial.number)+'.pt')
    return val_loss

study = optuna.create_study(directions=["minimize"])
study.optimize(objective, n_trials=10)#, timeout=300)
if losscoeff==1.0:
    losscoeff2 = int(losscoeff)
    read_and_proc.save_to_pickle(filepath+'vae/losscoeff_'+str(losscoeff2)+'/'+str(splitnum)+'/bestparams.pkt',study,'PICKLE')
else:
    read_and_proc.save_to_pickle(filepath+'vae/losscoeff_'+str(losscoeff)+'/'+str(splitnum)+'/bestparams.pkt',study,'PICKLE')
#if droprate==0.0:
#    droprate = int(droprate)
#    study = read_and_proc.depickle(filepath+'dropout/'+str(splitnum)+'/'+str(droprate)+'/bestparams.pkt')
#else:
#    study = read_and_proc.depickle(filepath+'dropout/'+str(splitnum)+'/'+str(droprate)+'/bestparams.pkt')
#study = read_and_proc.depickle('./store/dropout/'+str(expname)+'/'+str(droprate)+'/'+str(splitnum)+'/bestparams.pkt')

times = ['exp1a','exp1b','exp1c','exp1d','exp1e','exp1f','exp1g','exp1h','exp1i']
#times = ['exp1e','exp1f','exp1g','exp1h','exp1i']#,'exp1d','exp1e']
for i in range(len(times)):
    models,losses = [],[]
    model = vae.VAE(nummem[-2],nummem[-1],1,1,1,nummem)
    optimizers = [torch.optim.Adam(model.parameters(), lr=study.best_params['lr'])]#, optim.AdaBound(model.parameters(),lr=1e-7)] 1e-6 [torch.optim.Adam(model.parameters(),lr=0.5e-5),torch.optim.SGD(model.parameters(),lr=0.5e-5,momentum=0.8)]
    loss = torch.nn.L1Loss()
    for optimizer in optimizers:
        scheduler2 = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-8, max_lr=5e-5,cycle_momentum=False) #1e-9/1e-5
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',min_lr=1e-12)  #1e-18
        num_epochs = 1000*20#26
        #early_stopper = ts_models.EarlyStopping(patience=250, verbose=False, delta=1e-5, path='checkpoint.pt', trace_func=print)#EarlyStopper(patience=8, min_delta=1e-3)
        early_stopper = ts_models.EarlyStopping(patience=400, verbose=False, delta=1.5e-5, path='checkpoint.pt', trace_func=print)
        #variance_store = [varu,varv,varw,varth]
        #variance_store = [varu,varv,varth]
        model,loss,_ = vae.train_model(model=model,optimizer=optimizer,scheduler=[scheduler,scheduler2],numepochs=num_epochs,early_stopper=early_stopper,variance_store=None,\
                                         lossfunc=loss,train_loader=train_loader,val_loader=val_loader,test_loader=test_loader,l2_lambda=study.best_params['l2_lambda'],count=10,vaeloss_coeff=losscoeff)
        models.append(model)
        losses.append(loss)
    #torch.save(models, '../tmp/torch_try/ts/'+str(expname)+'/0/'+'models'+str(splitnum)+'_'+str(expname)+'3dnonln_1115_'+str(times[i])+'.pt')
    #read_and_proc.save_to_pickle('../tmp/torch_try/ts/'+str(expname)+'/0/'+'losses'+str(splitnum)+'_'+str(expname)+'3dnonln_1115_'+str(times[i])+'.pkt',losses,'PICKLE')
    if losscoeff==1.0:
        losscoeff2 = int(losscoeff)
        torch.save(models,filepath+'vae/losscoeff_'+str(losscoeff2)+'/'+str(splitnum)+'/modelstest'+str(splitnum)+'_vae_'+str(times[i])+'.pk')
        read_and_proc.save_to_pickle(filepath+'vae/losscoeff_'+str(losscoeff2)+'/'+str(splitnum)+'/lossestest'+str(splitnum)+'_vae_'+str(times[i])+'.pkt',losses,'PICKLE')
    else:
        torch.save(models,filepath+'vae/losscoeff_'+str(losscoeff)+'/'+str(splitnum)+'/modelstest'+str(splitnum)+'_vae_'+str(times[i])+'.pk')
        read_and_proc.save_to_pickle(filepath+'vae/losscoeff_'+str(losscoeff)+'/'+str(splitnum)+'/lossestest'+str(splitnum)+'_vae_'+str(times[i])+'.pkt',losses,'PICKLE')        
