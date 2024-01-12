import os,sys
sys.path.insert(1, '/work/FAC/FGSE/IDYST/tbeucler/default/freddy0218/TCGphy/2020_TC_CRF/dev/freddy0218/scikit/')
from tools import read_and_proc
import numpy as np
from tqdm.auto import tqdm
from sklearn.decomposition import IncrementalPCA
sys.path.insert(1, '../')
import read_stuff as read

def train_valid_test(expvarlist=None,validindex=None,testindex=None,concat='Yes'):
    X_valid, X_test = [expvarlist[i] for i in validindex], [expvarlist[i] for i in testindex]
    X_traint = expvarlist.copy()
    popindex = validindex+testindex
    X_train = [X_traint[i] for i in range(len(X_traint)) if i not in popindex]
    #assert len(X_train)==16, 'wrong train-valid-test separation!'
    if concat=='Yes':
        return np.concatenate([X_train[i] for i in range(len(X_train))],axis=0), np.concatenate([X_valid[i] for i in range(len(X_valid))],axis=0), np.concatenate([X_test[i] for i in range(len(X_test))],axis=0)
    else:
        return X_train, X_valid, X_test
    
class producePCA:
    def __init__(self,PCATYPE='varimax',n_comps=60):
        self.PCATYPE=PCATYPE
        self.n_comps=n_comps
    
    def fit_cheap_pca(self,n_batches=None,n_comps=None,var=None):
        from sklearn.decomposition import IncrementalPCA
        inc_pca = IncrementalPCA(n_components=n_comps)
        for X_batch in (np.array_split(var.data,n_batches)):
            inc_pca.partial_fit(X_batch)
        return inc_pca
        
    def fitPCA(self,arrays=None,arrayname=None,n_batches=10):
        """
        arrays: flat arrays to perform PCs
        arrayname: name of the variables
        axi: 2D or 3D
        """
        PCAdict = {}
        for ind,vnme in tqdm(enumerate(arrayname)):
            if self.PCATYPE=='varimax':
                try:
                    todo = arrays[ind]#-np.mean(arrays[ind])
                    PCAdict[vnme] = CustomPCA(n_components=self.n_comps,rotation='varimax').fit(todo)
                except:
                    sys.exit("Did not install R!")
            elif self.PCATYPE=='orig':
                PCAdict[vnme] = self.fit_cheap_pca(n_batches=10,n_comps=self.n_comps,var=arrays[ind])
        return PCAdict
        return None