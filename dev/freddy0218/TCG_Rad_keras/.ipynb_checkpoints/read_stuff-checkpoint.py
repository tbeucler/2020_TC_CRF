import glob,os,sys
import numpy as np
from tools.validation import r2_analysis
import pandas as pd
from tqdm.auto import tqdm
import gc
sys.path.insert(1, '/work/FAC/FGSE/IDYST/tbeucler/default/freddy0218/TCGphy/2020_TC_CRF/dev/freddy0218/scikit/')
from tools import derive_var,read_and_proc

def flatten(l):
    return [item for sublist in l for item in sublist]

def _get_exp_name(folderpath=None,splitnum=None,folder=2,TYPE='varimax'):
    if TYPE=='varimax':
        return sorted(glob.glob(folderpath+'varimaxpca/X/random/Xtrain*'))[splitnum][:-7].split('/')[-1][6:],sorted(glob.glob(folderpath+'varimaxpca/X/random/Xtrain*'))[splitnum][:-7].split('/')[-1][6:].split('_')
    elif TYPE=='orig':
        return sorted(glob.glob(folderpath+'pca/X/random/'+str(folder)+'/Xtrain*'))[splitnum][:-7].split('/')[-1][6:],sorted(glob.glob(folderpath+'pca/X/random/'+str(folder)+'/Xtrain*'))[splitnum][:-7].split('/')[-1][6:].split('_')
    elif TYPE=='keras':
        return sorted(glob.glob(folderpath+'keras/X/random/'+str(folder)+'/Xtrain*'))[splitnum][:-7].split('/')[-1][6:],sorted(glob.glob(folderpath+'keras/X/random/'+str(folder)+'/Xtrain*'))[splitnum][:-7].split('/')[-1][6:].split('_')
    elif TYPE=='fixTEST':
        return sorted(glob.glob(folderpath+'keras/Xnew/'+str(folder)+'/Xtrain*'))[splitnum][:-7].split('/')[-1][6:],sorted(glob.glob(folderpath+'keras/Xnew/'+str(folder)+'/Xtrain*'))[splitnum][:-7].split('/')[-1][6:].split('_')
        

def real_random(folderpath=None,index=None,folder=2,TYPE=None,yfolder=None):
    toextract = _get_exp_name(folderpath,index,folder,TYPE)[0]
    # X
    if TYPE=='varimax':
        Xtestpath,Xtrainpath,Xvalidpath = sorted(glob.glob(folderpath+'varimaxpca/X/random/*'+str(toextract)+'*'))
        yallpath = sorted(glob.glob(folderpath+'varimaxpca/y/random/*'+str(toextract)+'*'))
    elif TYPE=='orig':
        Xtestpath,Xtrainpath,Xvalidpath = sorted(glob.glob(folderpath+'pca/X/random/'+str(folder)+'/*'+str(toextract)+'*'))
        yallpath = sorted(glob.glob(folderpath+'pca/y/random/'+str(folder)+'/*'+str(toextract)+'*'))
    elif TYPE=='keras':
        Xtestpath,Xtrainpath,Xvalidpath = sorted(glob.glob(folderpath+'keras/X/random/'+str(folder)+'/*'+str(toextract)+'*'))
        yallpath = sorted(glob.glob(folderpath+'keras/y/random/'+str(yfolder)+'/*'+str(toextract)+'*'))
    elif TYPE=='fixTEST':
        Xtrainpath = sorted(glob.glob(folderpath+'keras/Xnew/'+str(folder)+'/Xtrain'+str(toextract)+'*'))
        Xvalidpath = sorted(glob.glob(folderpath+'keras/Xnew/'+str(folder)+'/Xvalid'+str(toextract)+'*'))
        Xtestpath = sorted(glob.glob(folderpath+'keras/Xnew/'+str(folder)+'/Xtest'+str(toextract)+'*'))
        #Xtestpath,Xtrainpath,Xvalidpath = sorted(glob.glob(folderpath+'keras/Xnew/'+str(folder)+'/*'+str(toextract)+'*'))
        yallpath = sorted(glob.glob(folderpath+'keras/ynew/'+str(yfolder)+'/allY'+str(toextract)+'*'))
    
    Xtest,Xtrain,Xvalid = [read_and_proc.depickle(obj) for obj in [Xtestpath[0],Xtrainpath[0],Xvalidpath[0]]]
    yall = read_and_proc.depickle(yallpath[0])
    return Xtest,Xtrain,Xvalid,yall

def real_random_y(folderpath=None,index=None,folder=2,TYPE=None,yfolder=None):
    toextract = _get_exp_name(folderpath,index,folder,TYPE)[0]
    # X
    if TYPE=='orig':
        yallpath = sorted(glob.glob(folderpath+'pca/y/random/'+str(folder)+'/*'+str(toextract)+'*'))
    elif TYPE=='keras':
        yallpath = sorted(glob.glob(folderpath+'keras/y/random/'+str(yfolder)+'/*'+str(toextract)+'*'))
    elif TYPE=='fixTEST':
        yallpath = sorted(glob.glob(folderpath+'keras/ynew/'+str(yfolder)+'/tsY'+str(toextract)+'*'))
    yall = read_and_proc.depickle(yallpath[0])
    return yall


from tools.validation import r2_analysis
class train_optimizedMLR:
    def __init__(self,folderpath=None,modelpath=None,subfoldername=None,ysubfoldername='rh',twoDthreeD='2D' or '3D'):
        self.pcapath=folderpath
        self.modelpath=modelpath
        if twoDthreeD=='2D':
            self.pcastore = read_and_proc.depickle(self.pcapath+'PCAdict2D.pkg')
            self.flatarray = read_and_proc.depickle(self.pcapath+'flatarrays2D.pkg')
        elif twoDthreeD=='3D':
            self.pcastore = read_and_proc.depickle(self.pcapath+'PCAdict3D.pkg')
            self.flatarray = read_and_proc.depickle(self.pcapath+'flatarrays3D.pkg')            
        self.subfoldername=subfoldername
        self.ysubfoldername=ysubfoldername
        self.twoDthreeD=twoDthreeD
        
    def read_Xy(self,subfolders='keras',num=33,needorig='No',onlyY='No' or 'Yes'):
        """
        Read in the processed PC loading time series
        """
        if onlyY=='No':
            Xtest,Xtrain,Xvalid = [],[],[]
            yall = []
            for i in tqdm(range(num)):
                temp1,temp2,temp3,temp4 = real_random(self.modelpath,i,self.subfoldername,subfolders,self.ysubfoldername)
                Xtest.append(temp1)
                Xtrain.append(temp2)
                Xvalid.append(temp3)
                yall.append(temp4)
                
            if needorig=='Yes':
                self.subfoldername=3
                yall_orig = []
                for i in tqdm(range(num)):
                    temp1,temp2,temp3,temp4 = real_random(self.modelpath,i,self.subfoldername,'orig')
                    yall_orig.append(temp4)
                return Xtrain,Xvalid,Xtest,yall,yall_orig
            else:
                return Xtrain,Xvalid,Xtest,yall
        elif onlyY=='Yes':
            yall = []
            for i in tqdm(range(num)):
                yall.append(real_random_y(self.modelpath,i,self.subfoldername,subfolders,self.ysubfoldername))
            return yall
                
    def delete_padding(self,inTS=None,outTS=None):
        output_nozero,input_nozero = [],[]
        if len(outTS.shape)>1:
            for i in range(len(outTS[:,0])):
                temp = outTS[i,:]
                tempin = inTS[i,:]
                if temp.all()==0:
                    continue
                else:
                    output_nozero.append(temp)
                    input_nozero.append(tempin)
            return input_nozero,output_nozero
        else:
            for i in range(len(outTS[:])):
                temp = outTS[i]
                tempin = inTS[i,:]
                if temp.all()==0:
                    continue
                else:
                    output_nozero.append(temp)
                    input_nozero.append(tempin)
            return input_nozero,output_nozero    
        
    def y_truth(self,divider=None,lti=24,num=33,withW=True,splitnum=None):
        if withW is True:
            temp = [r2_analysis.preproc_r2(self.flatarray,None,None)._back_to_exp(timeseries=self.flatarray[varname],divider=divider) for varname in ['u','v','w','theta']]
        else:
            temp = [r2_analysis.preproc_r2(self.flatarray,None,None)._back_to_exp(timeseries=self.flatarray[varname],divider=divider) for varname in ['u','v','theta']]
        train_realUV,valid_realUV,test_realUV = [],[],[]
        for ind,obj in tqdm(enumerate(splitnum)):#range(num)):#range(15)):#range(1)):
            try:
                tempindex = _get_exp_name(self.modelpath,obj,3,'orig')[1]
            except:
                tempindex = _get_exp_name('/work/FAC/FGSE/IDYST/tbeucler/default/freddy0218/TCGphy/2020_TC_CRF/dev/freddy0218/testML/output/haiyan/processed/intermediate/',obj,3,'orig')[1]                
            validindex,testindex = [int(tempindex[0]),int(tempindex[1])],[int(tempindex[2]),int(tempindex[3])]
            trainobj = r2_analysis.preproc_r2(self.flatarray,validindex,testindex).windrates_real(uvwheat=temp,LT=lti,category='train',withW=withW,twoDthreeD=self.twoDthreeD)
            validobj = r2_analysis.preproc_r2(self.flatarray,validindex,testindex).windrates_real(uvwheat=temp,LT=lti,category='valid',withW=withW,twoDthreeD=self.twoDthreeD)
            testobj = r2_analysis.preproc_r2(self.flatarray,validindex,testindex).windrates_real(uvwheat=temp,LT=lti,category='test',withW=withW,twoDthreeD=self.twoDthreeD)
            train_realUV.append(trainobj)
            valid_realUV.append(validobj)
            test_realUV.append(testobj)
        del trainobj,validobj,testobj
        gc.collect()
        return {'train':train_realUV,'valid':valid_realUV,'test':test_realUV}
    
    def _where_exp_MLRpred(self,splitnum=None,subfolders='fixTEST',divider=None,exp_index=None,LT=24):
        orig = np.r_[divider[0], np.diff(divider)]
        getindex = [int(obj) for obj in _get_exp_name(self.modelpath,splitnum,self.subfoldername,subfolders)[1]]
        if exp_index not in getindex:
            numexpout = sum([int(obj)<exp_index for obj in (getindex)])
            myindices = np.asarray([orig[i]-LT for i in range(len(orig)) if i not in getindex]).cumsum()
            return myindices[exp_index-numexpout-1],myindices[exp_index-numexpout],myindices[exp_index-numexpout]-myindices[exp_index-numexpout-1],'train'
        else:
            myidex = getindex.index(exp_index)
            if myidex<=1:
                category='valid'
                myindices = np.asarray([orig[i]-LT for i in range(len(orig)) if i in getindex[0:2]]).cumsum()
                if myidex==0:
                    return 0,myindices[0],myindices[0],category
                elif myidex==1:
                    return myindices[0],myindices[1],myindices[1]-myindices[0],category
            else:
                category='test'
                myindices = np.asarray([orig[i]-LT for i in range(len(orig)) if i in getindex[2:4]]).cumsum()
                if myidex==2:
                    return 0,myindices[0],myindices[0],category
                elif myidex==3:
                    return myindices[0],myindices[1],myindices[1]-myindices[0],category
                
    def where_exp_MLRpred(self,divider=None,num=40,expnum=10,LT=24):
        start,end,exp,size = [],[],[],[]
        for i in range(int(num)):
            temp1,temp2,temp3,temp4 = self._where_exp_MLRpred(splitnum=i,divider=divider,exp_index=expnum,LT=LT)
            start.append(temp1)
            end.append(temp2)
            exp.append(temp4)
            size.append(temp3)
            #except:
            #    start.append(None)
            #    end.append(None)
            #    exp.append(None)
            #    size.append(None)
        return pd.DataFrame.from_dict({'start':start,'end':end,'exp':exp,'size':size})