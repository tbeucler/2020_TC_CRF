import os,sys,gc
import numpy as np
import pickle
import torch
from tqdm.auto import tqdm
from skimage.metrics import structural_similarity as ssim
sys.path.insert(1, '/work/FAC/FGSE/IDYST/tbeucler/default/freddy0218/TCGphy/2020_TC_CRF/dev/freddy0218/scikit/')
from tools import derive_var,read_and_proc
from tools.mlr import mlr
from tools.preprocess import do_eof,preproc_maria,preproc_haiyan
sys.path.insert(2, '../')
import read_stuff as read
import glob

def output_pretrained_data(Xtrain=None,Xvalid=None,Xtest=None,yall=None,folderpath=None,folderpath2=None,splitnum=None):
    """──────────────────────────────────────────────────────────────────────────┐
    │ Which split will we be testing                     
    └──────────────────────────────────────────────────────────────────────────"""
    splitnum = splitnum
    """──────────────────────────────────────────────────────────────────────────┐
    │ Remove zero values                    
    └──────────────────────────────────────────────────────────────────────────"""
    X_totrain,y_totrain = read.train_optimizedMLR(folderpath,folderpath2,'rh','3D').delete_padding(Xtrain[splitnum]['lwswdtthuvwqv'],yall[splitnum][23][0])
    X_tovalid,y_tovalid = read.train_optimizedMLR(folderpath,folderpath2,'rh','3D').delete_padding(Xvalid[splitnum]['lwswdtthuvwqv'],yall[splitnum][23][1])
    X_totest,y_totest = read.train_optimizedMLR(folderpath,folderpath2,'rh','3D').delete_padding(Xtest[splitnum]['lwswdtthuvwqv'],yall[splitnum][23][2])
    
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
    return {'train':train_Xtensor,'val':val_Xtensor,'test':test_Xtensor},{'train':train_ytensor,'val':val_ytensor,'test':test_ytensor},{'train':train_data,'val':val_data,'test':test_data}

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

class analysis_trainedmodels:
    def __init__(self,folderpath=None,folderpath2=None,pcastore=None,divider=None,comps=None,Xtrain=None,Xvalid=None,Xtest=None,yall=None,linear='No' or 'Yes',nonlnnum=None):
        self.folderpath=folderpath
        self.folderpath2=folderpath2
        self.pcastore = pcastore
        self.divider = divider
        self.comps = comps
        self.Xtrain = Xtrain
        self.Xvalid = Xvalid
        self.Xtest = Xtest
        self.yall = yall
        self.linear = linear
        self.nonlnnum=nonlnnum
        
    def _get_ytruth(self,splitnum=None,leadtime=24):
        return read.train_optimizedMLR(self.folderpath,self.folderpath2,'rhorig','rhorig','3D').y_truth(divider=self.divider,lti=leadtime,num=33,withW=False,splitnum=[splitnum])
    
    def get_4layers(self,flatarray=None,vertlvs=10,layerindices=[0,2,5,8]):
        largearray = flatarray.reshape(flatarray.shape[0],vertlvs,360,int(flatarray.shape[1]/vertlvs/360))
        output = [largearray[:,i,...] for i in layerindices]
        del largearray
        gc.collect()
        return output
    
    def get_modelpred(self,splitnum=None,predcat='test'):
        models = output_trainedmodels(splitnum=splitnum,linear=self.linear,nonlnnum=self.nonlnnum)
        Xtensor,ytensor,data = output_pretrained_data(self.Xtrain,self.Xvalid,self.Xtest,self.yall,self.folderpath,self.folderpath2,splitnum)#output_processed(Xtrain,Xvalid,Xtest,yall,folderpath,folderpath2,split,linear,nonlnnum)
        predstore = []
        modelstore = []
        for i in range(len(models)):
            predstore.append(models[i][0](Xtensor[predcat]).detach().numpy())
            modelstore.append(models[i])
        return predstore,modelstore
    
    def pred_to_4d(self,pred=None,target='4layers' or 'all'):
        uout_store,vout_store,thout_store = [],[],[]
        for ind,obj in enumerate(pred):
            u = np.dot(obj[:,:self.comps[0]],self.pcastore['u'].components_[:self.comps[0]])
            v = np.dot(obj[:,self.comps[0]:self.comps[0]+self.comps[1]],self.pcastore['v'].components_[:self.comps[1]])
            th = np.dot(obj[:,self.comps[0]+self.comps[1]:self.comps[0]+self.comps[1]+self.comps[2]],self.pcastore['theta'].components_[:self.comps[2]])
            if target=='4layers':
                uout,vout,thout = self.get_4layers(u,10,[0,2,5,8]),self.get_4layers(v,10,[0,2,5,8]),self.get_4layers(th,10,[0,2,5,8])
            else:
                uout = u
                vout = u
                thout = th
            uout_store.append(uout)
            vout_store.append(vout)
            thout_store.append(thout)
            del u,v,th,uout,vout,thout
            gc.collect()
        return {'u':uout_store,'v':vout_store,'th':thout_store}
    
    def truth_to_4d(self,truth=None,target='4layers' or 'all',predcat='test'):
        utruth,vtruth,thtruth = truth[predcat][0]['du'],truth[predcat][0]['dv'],truth[predcat][0]['dth']
        return {'u':self.get_4layers(utruth,10,[0,2,5,8]), 'v':self.get_4layers(vtruth,10,[0,2,5,8]), 'th':self.get_4layers(thtruth,10,[0,2,5,8])}
    
    def _calc_ssmi(self,pred=None,truth=None,var='u',ANGLE=None):
        predVAR,truthVAR = pred[var],truth[var]
        modelsssim = []
        for ind in range(len(predVAR)): #models
            preds = predVAR[ind]
            ssimstore = []
            for layernum in range(len(preds)):
                predslayer,truthslayer = preds[layernum],truthVAR[layernum]
                #truthcart,predcart = [np.fliplr(np.flipud(read_and_proc.proc_tocart(truthslayer[i,...],ANGLE,True,False))) for i in (range(truthslayer.shape[0]))],\
                #[np.fliplr(np.flipud(read_and_proc.proc_tocart(predslayer[i,...],ANGLE,True,False))) for i in (range(truthslayer.shape[0]))]
                #ssimstore.append([ssim(truthcart[i],predcart[i],data_range=truthcart[i].max() - truthcart[i].min()) for i in (range(truthslayer.shape[0]))])
                #print(ssimstore)
                ssimstore.append([ssim(truthslayer[i,...],predslayer[i,...],data_range=truthslayer[i,...].max() - truthslayer[i,...].min()) for i in (range(truthslayer.shape[0]))])
            modelsssim.append(ssimstore)
        return modelsssim
            
    def calc_ssmi(self,predcat='test',ANGLES=None):
        ussim,vssim,thssim = [],[],[]
        for split in (range(33)):
            yTRUTH = self._get_ytruth(split,24)
            predstore,modelstore = self.get_modelpred(split,'test')
            dict_4layers = self.pred_to_4d(predstore,'4layers')
            truth_4layers = self.truth_to_4d(yTRUTH,'4layers',predcat)
            del yTRUTH,predstore
            gc.collect()
            
            ussim.append(self._calc_ssmi(dict_4layers,truth_4layers,'u',ANGLES))
            vssim.append(self._calc_ssmi(dict_4layers,truth_4layers,'v',ANGLES))
            thssim.append(self._calc_ssmi(dict_4layers,truth_4layers,'th',ANGLES))
        return ussim,vssim,thssim
        #return None
        
    def rank_choose_model(self,criteria='ssim',storeRESULTS=None,tokeep=None):
        if criteria=='ssim':
            storenonln = []
            for i in range(len(storeRESULTS)): #nonln
                ussim = storeRESULTS[i]
                storeperformance = []
                for j in range(len(ussim)): #model
                    storeperformance.append([(np.mean(obj1),np.mean(obj2),np.mean(obj3),np.mean(obj4)) for obj1,obj2,obj3,obj4 in (ussim[j])])
                storenonln.append(storeperformance)
            #print(len(storenonln[0][9]))
            del i,j
            
            numbers = np.asarray(storenonln)
            storenonln_ranked = np.zeros((numbers.shape[0],tokeep,2,numbers.shape[-1]))
            for i in range(numbers.shape[0]): #nonln
                for j in range(numbers.shape[-1]): #vertlayers
                    temp = list(zip(list(np.unravel_index(np.argsort(numbers[i,...,j],axis=None),numbers[i,...,j].shape))[0][-int(tokeep):],list(np.unravel_index(np.argsort(numbers[i,...,j],axis=None),numbers[i,...,j].shape))[1][-int(tokeep):]))
                    storenonln_ranked[i,...,j] = temp
            return storenonln_ranked
        
    def model_outweights(self,model=None):
        params,names = [],[]
        for name, param in model[0].named_parameters():
            if ".weight" not in name:
                continue
            else:
                params.append(param)
                names.append(name)
        return params, names