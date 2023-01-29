import glob,os,sys
import numpy as np
from tools.validation import r2_analysis
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
    
    Xtest,Xtrain,Xvalid = [read_and_proc.depickle(obj) for obj in[Xtestpath,Xtrainpath,Xvalidpath]]
    yall = read_and_proc.depickle(yallpath[0])
    return Xtest,Xtrain,Xvalid,yall


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
        
    def read_Xy(self,num=33,needorig='No'):
        """
        Read in the processed PC loading time series
        """
        Xtest,Xtrain,Xvalid = [],[],[]
        yall = []
        for i in tqdm(range(num)):
            temp1,temp2,temp3,temp4 = real_random(self.modelpath,i,self.subfoldername,'keras',self.ysubfoldername)
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
    
    def delete_padding(self,inTS=None,outTS=None):
        output_nozero,input_nozero = [],[]
        for i in range(len(outTS[:,0])):
            temp = outTS[i,:]
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