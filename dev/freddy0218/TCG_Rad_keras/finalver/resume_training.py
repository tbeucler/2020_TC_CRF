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
import read_stuff as read
import ts_models
import glob,os
import matplotlib.pyplot as plt
import ts_models,analysis

splitnum=int(str(sys.argv[1]))
droprate=float(str(sys.argv[2]))
nonln_num=int(str(sys.argv[3]))

if droprate==0.0:
    droprate=int(0)

def prepare_tensors(filepath=None,splitnum=None,explag=23):
    Xtrain = read_and_proc.depickle(glob.glob(filepath+'X/'+str(splitnum)+'/*intensity*')[0])['uvwthhdialwsw_inten']['train']
    Xvalid = read_and_proc.depickle(glob.glob(filepath+'X/'+str(splitnum)+'/*intensity*')[0])['uvwthhdialwsw_inten']['valid']
    Xtest = read_and_proc.depickle(glob.glob(filepath+'X/'+str(splitnum)+'/*intensity*')[0])['uvwthhdialwsw_inten']['test']
    
    ytrain = read_and_proc.depickle(glob.glob(filepath+'y/'+str(splitnum)+'/*intensity*')[0])['train'][explag]
    yvalid = read_and_proc.depickle(glob.glob(filepath+'y/'+str(splitnum)+'/*intensity*')[0])['valid'][explag]
    ytest = read_and_proc.depickle(glob.glob(filepath+'y/'+str(splitnum)+'/*intensity*')[0])['test'][explag]
    
    X_totrain, y_totrain = read.delete_padding(Xtrain,ytrain)
    X_tovalid, y_tovalid = read.delete_padding(Xvalid,yvalid)
    X_totest, y_totest = read.delete_padding(Xtest,ytest)
    print(np.asarray(X_totest).shape)
    
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
                                                                                   
    return train_data,val_data,test_data

class resume_training:
    def __init__(self,splitnum=None,droprate=None,nonln_num=None,timelag=None,batch_size=None,num_workers=2,brchindex=None):
        self.splitnum=splitnum
        self.droprate=droprate
        self.nonln_num=nonln_num
        self.timelag = timelag
        self.batch_size = batch_size
        self.num_workers=2
        self.brchindex=brchindex
        
    def get_data(self,filepath=None):
        train_data,val_data,test_data = prepare_tensors(filepath,self.splitnum,self.timelag)
        train_loader = torch.utils.data.DataLoader(dataset=train_data,batch_size=self.batch_size,shuffle=True)
        val_loader = torch.utils.data.DataLoader(dataset=val_data,batch_size=self.batch_size,shuffle=False)
        test_loader = torch.utils.data.DataLoader(dataset=test_data,batch_size=self.batch_size,shuffle=False)
        return train_loader,val_loader,test_loader
    
    def continue_training(self,datafilepath='./maria_store/',savefilepath='./maria_store/dropout_corr/',exp='e',scheduler_lr=[1e-14,5e-10]):
        train_loader,val_loader,_ = self.get_data(datafilepath)
        if self.nonln_num==0:
            study = read_and_proc.depickle(savefilepath+str(splitnum)+'/'+str(droprate)+'/'+'bestparams.pkt')
            original_model = ts_models.OptimMLR_lwsw_3D_ts_dropout2(self.droprate,self.brchindex)#[0,50,26,50,50,50,10,10])
        else:
            study = read_and_proc.depickle(savefilepath+str(splitnum)+'/'+str(droprate)+'/'+'bestparams.pkt')
            original_model = ts_models.OptimMLR_lwsw_3D_ts_dropout2_nonln(self.droprate,self.brchindex,self.nonln_num)#[0,50,26,50,50,50,10,10],self.nonln_num)
        #######################################################################################################################################
        # Transfer state dict
        pretrained_model = torch.load(savefilepath+str(self.splitnum)+'/'+str(self.droprate)+'/modelstest'+str(self.splitnum)+'_lwswts_1115_exp1'+str(exp)+'.pt')[0]
        model_dict = original_model.state_dict()
        pretrained_dict = pretrained_model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        original_model.load_state_dict(model_dict)
        #######################################################################################################################################
        #######################################################################################################################################
        optimizer = torch.optim.Adam(original_model.parameters(), lr=study.best_params['lr'])
        lossfunc = torch.nn.L1Loss()
        #scheduler2 = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-16, max_lr=5e-10,cycle_momentum=False) #1e-9/1e-5
        scheduler2 = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=scheduler_lr[0], max_lr=scheduler_lr[1],cycle_momentum=False) #1e-9/1e-5
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',min_lr=1e-20)
        #######################################################################################################################################
        
        lowest_val_loss = float('inf')
        best_model = None
        schedulerCY,schedulerLS = scheduler2,scheduler
        train_losses = []
        val_losses = []
        
        for epoch in tqdm(range(20000)):
            original_model.train()
            train_loss = 0
            # Training loop here
            for features, labels in train_loader:
                optimizer.zero_grad()
                prediction = original_model(features)
                batch_loss = lossfunc(prediction, labels.unsqueeze(1))#loss_func(prediction, labels)
                batch_loss.backward()
                optimizer.step()
                schedulerCY.step()
                
                train_loss += batch_loss.item()    
            train_loss = train_loss / len(train_loader)
            train_losses.append(train_loss)

            # Validation loop
            original_model.eval()
            with torch.no_grad():
                val_loss = 0
                for features, labels in val_loader:
                    pred = original_model(features)
                    batch_loss = lossfunc(pred, labels.unsqueeze(1))
                    val_loss+=batch_loss.item()
            
                val_loss = val_loss / len(val_loader)
                val_losses.append(val_loss)

            # Check if the current model has the lowest validation loss
            if val_loss < lowest_val_loss:
                lowest_val_loss = val_loss
                best_model = original_model.state_dict()
    
            torch.save(best_model, savefilepath+str(splitnum)+'/'+str(droprate)+'/modelstest'+str(splitnum)+'_lwswts_1115_exp1'+str(exp)+'_best.pt')
            read_and_proc.save_to_pickle(savefilepath+str(splitnum)+'/'+str(droprate)+'/lossestest'+str(splitnum)+'_lwswts_1115_exp1'+str(exp)+'_best.pkt',{'train':train_losses,'val':val_losses},'PICKLE')
        return None
    
filepath = './haiyan_store/dropout_old/nonln_'+str(nonln_num)+'/'
brchindex = [0,50,38,50,8,50,20,20]
for exp in ['a','b','c','d','e','f','g','h','i']:
    resume_training(splitnum,droprate,nonln_num,23,9,2,brchindex).continue_training(datafilepath='./haiyan_store/',savefilepath=filepath,\
                                                                exp=exp,scheduler_lr=[1e-14,5e-10])
