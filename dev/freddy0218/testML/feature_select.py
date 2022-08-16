from sklearn.feature_selection import SelectorMixin
from sklearn.metrics import r2_score
from sklearn.base import BaseEstimator, MetaEstimatorMixin, clone
from sklearn.model_selection import cross_val_score
from tools.mlr import mlr,proc_mlrfcst,maria_IO
from tqdm.auto import tqdm
import numpy as np
import gc

class forwardfeatureadder(BaseEstimator,SelectorMixin,MetaEstimatorMixin):
    """Transformer to add feature at a sequential order
    Parameters:
    estimator: Regression model
    n_features_to_select: number of features to add to the model
    cv: how many folds would we want during cross-validation
    n_jobs: Parallelization
    startfeatures: Features we would like to include in the model without cross-validation [we do this to accentuate the role of heating]
    
    Output:
    self instance
    """
    def __init__(self,estimator,n_features_to_select=None,cv=5,n_jobs=None,startfeatures=None,PCAdict=None,Afdict=None,numcomp=None,LT=None,optigoal='surface',Xsurf=None,Ysurf=None):
        self.estimator = estimator
        self.n_features_to_select = n_features_to_select
        self.cv = cv
        self.n_jobs = n_jobs
        self.startfeatures = startfeatures
        self.PCAdict = PCAdict
        self.Afdict = Afdict
        self.numcomp = numcomp
        self.LT = LT
        self.optigoal = optigoal
        self.Xsurf=Xsurf
        self.Ysurf=Ysurf
    
    def get_real_winds(self):
        temp1,temp2,temp3,temp4 = proc_mlrfcst.retrieve_cartesian(PCA_dict=self.PCAdict,Af_dict=self.Afdict,numcomp=self.numcomp,LT=self.LT,
                            forecastPC=None).windrates_real(LT=self.LT)
        return temp1,temp2,temp3,temp4
    
    def convert_forecast_winds(self,yforecast=None):
        try:
            temp1,temp2,temp3,temp4 = proc_mlrfcst.retrieve_cartesian(PCA_dict=self.PCAdict,Af_dict=self.Afdict,numcomp=self.numcomp,LT=self.LT,
                                                                      forecastPC=yforecast).output_reshapeRECON(forecast_eig=yforecast[int(self.LT[i]-1)])
            return temp1,temp2,temp3,temp4
        except:
            temp1,temp2 = proc_mlrfcst.retrieve_cartesian(PCA_dict=self.PCAdict,Af_dict=self.Afdict,numcomp=self.numcomp,LT=self.LT,
                                                                      forecastPC=yforecast,target='surface').output_reshapeRECON(forecast_eig=yforecast)        
            return temp1,temp2
    
    def fit(self, X,y=None):
        """Learn features to select from X.
        X (n_samples,n_features): Training vectors
        Y (n_samples): Target values
        """
        # Define basic settings
        n_features = X.shape[1]
        current_mask = np.zeros(shape=n_features,dtype=bool)
        for index in self.startfeatures:
            current_mask[index] = True
        n_iteractions = self.n_features_to_select
        
        # Do forward selection
        addinput,r2 = [],[]
        clone_estimator = clone(self.estimator)
        for _ in range(n_iteractions):
            #new_feature_idx,r2t = self.get_best_new_feature_R2based(clone_estimator,X,y,current_mask)
            new_feature_idx = self.get_best_new_feature(clone_estimator,X,y,current_mask)
            #r2.append(r2t)
            current_mask[new_feature_idx] = True
            addinput.append(current_mask)
        
        self.support_ = current_mask
        self.new_feature = new_feature_idx
        self.r2 = r2
        return self
    
    def get_best_new_feature(self,estimator,X,y,current_mask):
        candidate_feature_indices = np.flatnonzero(~current_mask)
        scores={}
        for feature_idx in candidate_feature_indices:
            candidate_mask = current_mask.copy()
            candidate_mask[feature_idx] = True
            
            # Add a new feature
            X_new = X[:,candidate_mask]
            # Improvement
            scores[feature_idx] = cross_val_score(estimator,X_new,y,cv=self.cv,scoring=None,n_jobs=self.n_jobs).mean()
        return max(scores,key=lambda feature_idx: scores[feature_idx])
    
    #--------------------------------------------------------------
    # To do -> Add featureselector based on r2
    # Candidate mask -> Xnew
    # fit linear model with (Xnew,y)
    # {output r2 term [time consideration => target: surface u/v]}...repeat for all u/v/w/theta members
    # get component index that results in best r2 score
    # --------[[Exit loops when r2 reaches 0.75?]]-----------------
    # add to mask during fitting 
    #---------------------------------------------------------------------------------------------------
    def get_best_new_feature_R2based(self,estimator,X,y,current_mask):             
        candidate_feature_indices = np.flatnonzero(~current_mask)
        scores={}
        for feature_idx in candidate_feature_indices:
            candidate_mask = current_mask.copy()
            candidate_mask[feature_idx] = True
            # Add a new feature
            X_new = X[:,candidate_mask]
            # Improvement
            LDTME = np.linspace(0,44,45)+1
            y_forecast = mlr.model_fitpredict(X_new,y,estimator,LDTME).modelfit(singleLT=True)[0].predict(X_new)
            #################################################################################################################################################################################################################
            # Forecast winds
            #################################################################################################################################################################################################################
            if self.optigoal=='surface':
                teMP1,teMP2 = self.convert_forecast_winds(y_forecast)
                teMP1s,teMP2s = (teMP1.reshape(teMP1.shape[0],39,360,167)[:,0,:,:]).reshape(teMP1.shape[0],360*167),(teMP2.reshape(teMP2.shape[0],39,360,167)[:,0,:,:]).reshape(teMP2.shape[0],360*167)
                del teMP1,teMP2
                gc.collect()
                
                scores[feature_idx] = r2_score(np.concatenate((self.Xsurf,self.Ysurf),axis=0),np.concatenate((teMP1s,teMP2s),axis=0))
            else:
                teMP1,teMP2,teMP3,teMP4 = self.convert_forecast_winds(y_forecast)                
                scores[feature_idx] = r2_score(np.concatenate((reteMP1,reteMP2,reteMP3,reteMP4),axis=0),np.concatenate((teMP1,teMP2,teMP3,teMP4),axis=0))
        return max(scores,key=lambda feature_idx: scores[feature_idx]),max(scores.values())
    
    def _get_support_mask(self):
        return self.support_
    
class FFW_process:
    def __init__(self,real_u=None,real_v=None,PCAdict=None,Af_dict=None,numcomp=None,LT=None,target=None,case='Maria'):
        self.real_u = real_u
        self.real_v = real_v
        self.PCAdict = PCAdict
        self.Afdict = Af_dict
        self.numcomp = numcomp
        self.LT = LT
        self.target=target
        self.case=case

    def conversion_predictPC(self,yforecast=None,mshpe=[39,360,167]):
        if self.target=='surface':
            t1,t2 = proc_mlrfcst.retrieve_cartesian(PCA_dict=self.PCAdict,Af_dict=self.Afdict,numcomp=self.numcomp,LT=self.LT,forecastPC=yforecast,target=self.target).output_reshapeRECON(forecast_eig=yforecast)
            return (t1.reshape(t1.shape[0],mshpe[0],mshpe[1],mshpe[2])[:,0,:,:]).reshape(t1.shape[0],mshpe[1]*mshpe[2]),(t2.reshape(t2.shape[0],mshpe[0],mshpe[1],mshpe[2])[:,0,:,:]).reshape(t2.shape[0],mshpe[1]*mshpe[2])
        else:
            return None
        
    def output_r2(self,FFWmodels=None,reducedX=None):
        yf = [models.predict(Xns) for (models,Xns) in zip(FFWmodels[:],reducedX[:])]
        mlr_r2 = []
        for i in tqdm(range(20)):#len(yf))):
            if self.case=='Maria':
                temp1,temp2 = self.conversion_predictPC(yforecast=yf[i],mshpe=[39,360,167])
            mlr_r2.append(r2_score(np.concatenate((self.real_u,self.real_v),axis=0),np.concatenate((temp1,temp2),axis=0)))
            del temp1,temp2
            gc.collect()
        return mlr_r2
    
    