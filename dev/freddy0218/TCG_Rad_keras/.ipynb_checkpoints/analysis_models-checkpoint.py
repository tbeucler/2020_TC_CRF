import glob
import os,sys,gc
import numpy as np
import pickle
import torch
from tqdm.auto import tqdm
import pandas as pd

sys.path.insert(1, '/work/FAC/FGSE/IDYST/tbeucler/default/freddy0218/TCGphy/2020_TC_CRF/dev/freddy0218/scikit/')
from tools import derive_var,read_and_proc
from tools.mlr import mlr
from tools.preprocess import do_eof,preproc_maria,preproc_haiyan
from tools.validation import r2_analysis
sys.path.insert(2, '../')
import read_stuff as read
from tqdm.auto import tqdm

def output_trainedmodels(path=None,splitnum=None,linear='Yes',nonlnnum=None):
    if linear=='Yes':
        modelpaths = sorted(glob.glob('../tmp/torch_try/1115_3d/'+str(splitnum)+'/models*'))
        model = [torch.load(obj) for obj in modelpaths]
    elif linear=='No':
        modelpaths = sorted(glob.glob('../tmp/torch_try/pytorch_3d_nonlinear/'+str(nonlnnum)+'/models'+str(splitnum)+'_'+'*'))
        model = [torch.load(obj) for obj in modelpaths]        
    return model

def output_trainedmodels_ffs(path=None,splitnum=None,nonlnnum=None):
    modelpaths = sorted(glob.glob(path+str(nonlnnum)+'/models'+str(splitnum)+'_'+'*'))
    model = [torch.load(obj) for obj in modelpaths]        
    return model

def output_pretrained_data(Xtrain=None,Xvalid=None,Xtest=None,yall=None,folderpath=None,folderpath2=None,splitnum=None,exp='lwswdtthuvw'):
    """──────────────────────────────────────────────────────────────────────────┐
    │ Which split will we be testing                     
    └──────────────────────────────────────────────────────────────────────────"""
    splitnum = splitnum
    """──────────────────────────────────────────────────────────────────────────┐
    │ Remove zero values                    
    └──────────────────────────────────────────────────────────────────────────"""
    X_totrain,y_totrain = read.train_optimizedMLR(folderpath,folderpath2,'rh','3D').delete_padding(Xtrain[splitnum][exp],yall[splitnum][23][0])
    X_tovalid,y_tovalid = read.train_optimizedMLR(folderpath,folderpath2,'rh','3D').delete_padding(Xvalid[splitnum][exp],yall[splitnum][23][1])
    X_totest,y_totest = read.train_optimizedMLR(folderpath,folderpath2,'rh','3D').delete_padding(Xtest[splitnum][exp],yall[splitnum][23][2])
    
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

from skimage.metrics import structural_similarity as ssim
import pandas as pd
from sklearn.metrics import r2_score
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
    
    def _get_tensors(self,splitnum=None,exp='lwswdtthuvw'):
        return output_pretrained_data(self.Xtrain,self.Xvalid,self.Xtest,self.yall,self.folderpath,self.folderpath2,splitnum,exp)
    
    def get_4layers(self,flatarray=None,vertlvs=10,layerindices=[0,2,5,8]):
        largearray = flatarray.reshape(flatarray.shape[0],vertlvs,360,int(flatarray.shape[1]/vertlvs/360))
        output = [largearray[:,i,...] for i in layerindices]
        del largearray
        gc.collect()
        return output
    
    def get_modelpred(self,splitnum=None,predcat='test',exp='lwswdtthuvw'):
        models = output_trainedmodels(splitnum=splitnum,linear=self.linear,nonlnnum=self.nonlnnum)
        Xtensor,ytensor,data = output_pretrained_data(self.Xtrain,self.Xvalid,self.Xtest,self.yall,self.folderpath,self.folderpath2,splitnum,exp)#output_processed(Xtrain,Xvalid,Xtest,yall,folderpath,folderpath2,split,linear,nonlnnum)
        predstore = []
        modelstore = []
        for i in range(len(models)):
            predstore.append(models[i][0](Xtensor[predcat]).detach().numpy())
            modelstore.append(models[i])
        return predstore,modelstore
    
    def get_modelpred_ffs(self,path=None,splitnum=None,predcat='test',exp='lwswdtthuvw'):
        models = output_trainedmodels_ffs(path=path,splitnum=splitnum,nonlnnum=self.nonlnnum)
        Xtensor,ytensor,data = output_pretrained_data(self.Xtrain,self.Xvalid,self.Xtest,self.yall,self.folderpath,self.folderpath2,splitnum,exp)#output_processed(Xtrain,Xvalid,Xtest,yall,folderpath,folderpath2,split,linear,nonlnnum)
        predstore = []
        modelstore = []
        for i in range(len(models)):
            predstore.append(models[i][0](Xtensor[predcat]).detach().numpy())
            modelstore.append(models[i])
        return predstore,modelstore
    
    def get_pd_preds(self,model=None,alt_Xtensor=None):
        return models[0](alt_Xtensor[predcat]).detach().numpy()
    
    def onepred_to_4d(self,pred=None,target='4layers' or 'all'):
        u = np.dot(pred[:,:self.comps[0]],self.pcastore['u'].components_[:self.comps[0]])
        v = np.dot(pred[:,self.comps[0]:self.comps[0]+self.comps[1]],self.pcastore['v'].components_[:self.comps[1]])
        th = np.dot(pred[:,self.comps[0]+self.comps[1]:self.comps[0]+self.comps[1]+self.comps[2]],self.pcastore['theta'].components_[:self.comps[2]])
        if target=='4layers':
            uout,vout,thout = self.get_4layers(u,10,[0,2,5,8]),self.get_4layers(v,10,[0,2,5,8]),self.get_4layers(th,10,[0,2,5,8])
        else:
            uout = u
            vout = v
            thout = th
        return {'u':uout,'v':vout,'th':thout}
    
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
                vout = v
                thout = th
            uout_store.append(uout)
            vout_store.append(vout)
            thout_store.append(thout)
            del u,v,th,uout,vout,thout
            gc.collect()
        return {'u':uout_store,'v':vout_store,'th':thout_store}
    
    def truth_to_4d(self,truth=None,target='4layers' or 'all',predcat='test'):
        utruth,vtruth,thtruth = truth[predcat][0]['du'],truth[predcat][0]['dv'],truth[predcat][0]['dth']
        if target=='4layers':
            return {'u':self.get_4layers(utruth,10,[0,2,5,8]), 'v':self.get_4layers(vtruth,10,[0,2,5,8]), 'th':self.get_4layers(thtruth,10,[0,2,5,8])}
        elif target=='all':
            return {'u':utruth,'v':vtruth,'th':thtruth}
    
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
            
    def calc_ssmi(self,predcat='test',ANGLES=None,exp='lwswdtthuvw'):
        ussim,vssim,thssim = [],[],[]
        for split in (range(33)):
            yTRUTH = self._get_ytruth(split,24)
            predstore,modelstore = self.get_modelpred(split,'test',exp)
            dict_4layers = self.pred_to_4d(predstore,'4layers')
            truth_4layers = self.truth_to_4d(yTRUTH,'4layers',predcat)
            del yTRUTH,predstore
            gc.collect()
            
            ussim.append(self._calc_ssmi(dict_4layers,truth_4layers,'u',ANGLES))
            vssim.append(self._calc_ssmi(dict_4layers,truth_4layers,'v',ANGLES))
            thssim.append(self._calc_ssmi(dict_4layers,truth_4layers,'th',ANGLES))
        return ussim,vssim,thssim
        #return None

    def _calc_r2(self,pred=None,truth=None,var='u',ANGLE=None):
        predVAR,truthVAR = pred[var],truth[var]
        modelr2 = []
        for ind in range(len(predVAR)): #models
            preds = predVAR[ind]
            modelr2.append(r2_score(truthVAR.transpose(),preds.transpose()))
        return modelr2
    
    def calc_r2(self,predcat='test',ANGLES=None,exp='lwswdtthuvw'):
        ur2,vr2,thr2 = [],[],[]
        for split in (range(33)):
            yTRUTH = self._get_ytruth(split,24)
            predstore,modelstore = self.get_modelpred(split,'test',exp)
            dict_all = self.pred_to_4d(predstore,'all')
            truth_all = self.truth_to_4d(yTRUTH,'all',predcat)
            del yTRUTH,predstore
            gc.collect()
            
            ur2.append(self._calc_r2(dict_all,truth_all,'u',ANGLES))
            vr2.append(self._calc_r2(dict_all,truth_all,'v',ANGLES))
            thr2.append(self._calc_r2(dict_all,truth_all,'th',ANGLES))
            del dict_all,truth_all
            gc.collect()
        return ur2,vr2,thr2

    def calc_r2_ffs(self,path=None,predcat='test',ANGLES=None,exp='lwswdtthuvw'):
        ur2,vr2,thr2 = [],[],[]
        for split in (range(33)):
            yTRUTH = self._get_ytruth(split,24)
            predstore,modelstore = self.get_modelpred_ffs(path,split,'test',exp)
            dict_all = self.pred_to_4d(predstore,'all')
            truth_all = self.truth_to_4d(yTRUTH,'all',predcat)
            del yTRUTH,predstore
            gc.collect()
            
            ur2.append(self._calc_r2(dict_all,truth_all,'u',ANGLES))
            vr2.append(self._calc_r2(dict_all,truth_all,'v',ANGLES))
            thr2.append(self._calc_r2(dict_all,truth_all,'th',ANGLES))
            del dict_all,truth_all
            gc.collect()
        return ur2,vr2,thr2
        
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
        for name, param in bestmodel[0].named_parameters():
            if ".weight" not in name:
                continue
            else:
                params.append(param)
                names.append(name)
        return params, names

    def _where_exp_MLRpred(self,splitnum=None,exp_index=None,LT=24):
        orig = np.r_[self.divider[0], np.diff(self.divider)]
        getindex = [int(obj) for obj in read._get_exp_name('/work/FAC/FGSE/IDYST/tbeucler/default/freddy0218/TCGphy/2020_TC_CRF/dev/freddy0218/testML/output/haiyan/processed/intermediate/',splitnum,3,'orig')[1]]
        
        if exp_index not in getindex:
            numexpout = sum([int(obj)<exp_index for obj in (getindex)])
            myindices = np.asarray([orig[i]-LT for i in range(len(orig)) if i not in getindex]).cumsum()
            if exp_index==0:
                return 0,myindices[exp_index-numexpout],myindices[exp_index-numexpout]-0,'train',getindex
            else:
                return myindices[exp_index-numexpout-1],myindices[exp_index-numexpout],myindices[exp_index-numexpout]-myindices[exp_index-numexpout-1],'train',getindex
        else:
            myidex = getindex.index(exp_index)
            if myidex<=1:
                category='valid'
                myindices = np.asarray([orig[i]-LT for i in range(len(orig)) if i in getindex[0:2]]).cumsum()
                if myidex==0:
                    return 0,myindices[0],myindices[0],category,getindex
                elif myidex==1:
                    return myindices[0],myindices[1],myindices[1]-myindices[0],category,getindex
            else:
                category='test'
                myindices = np.asarray([orig[i]-LT for i in range(len(orig)) if i in getindex[2:4]]).cumsum()
                if myidex==2:
                    return 0,myindices[0],myindices[0],category,getindex
                elif myidex==3:
                    return myindices[0],myindices[1],myindices[1]-myindices[0],category,getindex

    def where_exp_MLRpred(self,expnum=10,LT=24):
        start,end,exp,size,expname = [],[],[],[],[]
        for i in range(33):
            temp1,temp2,temp3,temp4,temp5 = self._where_exp_MLRpred(i,expnum,LT)
            start.append(temp1)
            end.append(temp2)
            exp.append(temp4)
            size.append(temp3)
            expname.append(temp5)
            #except:
            #    start.append(None)
            #    end.append(None)
            #    exp.append(None)
            #    size.append(None)
        return pd.DataFrame.from_dict({'start':start,'end':end,'exp':exp,'size':size,'splitinfo':expname})
    
    def get_multiple_r2(self,path=None,predcat='test',ANGLE=None,exp='lwswdtthuvw',totalsplit=None,modelsplit=None):
        # get models with one split
        models = [obj[0] for obj in self.get_modelpred_ffs(path,modelsplit,predcat,exp)[1]]
        # get preds
        ur2,vr2,thr2 = [],[],[]
        temp = []
        for ind,obj in enumerate(totalsplit):
            Xtensor = output_pretrained_data(self.Xtrain,self.Xvalid,self.Xtest,self.yall,self.folderpath,self.folderpath2,splitnum=obj,exp=exp)[0]
            modelpreds = [models[i](Xtensor[predcat]).detach().numpy() for i in range(len(models))]
            yTRUTH = self._get_ytruth(obj,24)
            dict_alls = self.pred_to_4d(modelpreds,'all')
            truth_all = self.truth_to_4d(yTRUTH,'all',predcat)
            del yTRUTH,modelpreds
            gc.collect()
            
            ur2.append(self._calc_r2(dict_alls,truth_all,'u',ANGLE))
            vr2.append(self._calc_r2(dict_alls,truth_all,'v',ANGLE))
            thr2.append(self._calc_r2(dict_alls,truth_all,'th',ANGLE))
            del dict_alls,truth_all
            gc.collect()
        return ur2,vr2,thr2
            
        
    def calc_r2_ffs(self,path=None,predcat='test',ANGLES=None,exp='lwswdtthuvw'):
        ur2,vr2,thr2 = [],[],[]
        for split in (range(33)):
            yTRUTH = self._get_ytruth(split,24)
            predstore,modelstore = self.get_modelpred_ffs(path,split,'test',exp)
            dict_all = self.pred_to_4d(predstore,'all')
            truth_all = self.truth_to_4d(yTRUTH,'all',predcat)
            del yTRUTH,predstore
            gc.collect()
            
            ur2.append(self._calc_r2(dict_all,truth_all,'u',ANGLES))
            vr2.append(self._calc_r2(dict_all,truth_all,'v',ANGLES))
            thr2.append(self._calc_r2(dict_all,truth_all,'th',ANGLES))
            del dict_all,truth_all
            gc.collect()
        return ur2,vr2,thr2        