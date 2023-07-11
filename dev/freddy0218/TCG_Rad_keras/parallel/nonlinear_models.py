import os,sys,gc
import numpy as np
import pickle
import torch
from livelossplot import PlotLosses
from tqdm.auto import tqdm

class OptimMLR_all_3D_lwswu(torch.nn.Module):
    def __init__(self,num_nonlinear):
        #super(OptimMLR_all_2D, self).__init__()
        super(OptimMLR_all_3D_lwswu, self).__init__()
        ############################################################
        # Input channels
        ############################################################
        brchsize = [50,20,20]
        self.dense1 = torch.nn.Linear(brchsize[0], 1)
        self.dense2 = torch.nn.Linear(brchsize[1], 1)
        self.dense3 = torch.nn.Linear(brchsize[2], 1)
        ############################################################
        # Nonlinear layers
        ############################################################
        self.nonln = torch.nn.ModuleList([
            torch.nn.Sequential(torch.nn.Linear(len(brchsize),len(brchsize)),
                                torch.nn.LeakyReLU()) for i in range(num_nonlinear)])

        #for i in range(num_nonlinear):
        #    self.nonln.append(torch.nn.Linear(7,7))
        #    self.nonln.append(torch.nn.LeakyReLU())
        #self.nonln = torch.nn.
        ############################################################
        # Final Dense Layer
        ############################################################
        self.denseout = torch.nn.Linear(len(brchsize),96)#106)

    def forward(self,X):
        brchindex = list(np.asarray([0,50,38,91,8,82,20,20]).cumsum())
        X_u, X_v, X_w, X_th = X[:,brchindex[0]:brchindex[1]],X[:,brchindex[1]:brchindex[2]],X[:,brchindex[2]:brchindex[3]],X[:,brchindex[3]:brchindex[4]]
        X_hdia, X_lw, X_sw = X[:,brchindex[4]:brchindex[5]],X[:,brchindex[5]:brchindex[6]],X[:,brchindex[6]:brchindex[7]]
        ############################################################
        # Optimal PC layer
        ############################################################
        bestu = self.dense1(X_u)
        bestlw = self.dense2(X_lw)
        bestsw = self.dense3(X_sw)
        ############################################################
        # Concat
        ############################################################
        bestPC = torch.cat((bestu,bestlw,bestsw),1)
        bestPC_proc = bestPC
        ############################################################
        # Nonlinear layer
        ############################################################
        for nonln in self.nonln:
            bestPC_proc = nonln(bestPC_proc)
        ############################################################
        # Prediction layer
        ############################################################
        outpred = self.denseout(bestPC_proc)
        return outpred

class OptimMLR_all_3D_lwswv(torch.nn.Module):
    def __init__(self,num_nonlinear):
        #super(OptimMLR_all_2D, self).__init__()
        super(OptimMLR_all_3D_lwswv, self).__init__()
        ############################################################
        # Input channels
        ############################################################
        brchsize = [38,20,20]
        self.dense1 = torch.nn.Linear(brchsize[0], 1)
        self.dense2 = torch.nn.Linear(brchsize[1], 1)
        self.dense3 = torch.nn.Linear(brchsize[2], 1)
        ############################################################
        # Nonlinear layers
        ############################################################
        self.nonln = torch.nn.ModuleList([
            torch.nn.Sequential(torch.nn.Linear(len(brchsize),len(brchsize)),
                                torch.nn.LeakyReLU()) for i in range(num_nonlinear)])

        #for i in range(num_nonlinear):
        #    self.nonln.append(torch.nn.Linear(7,7))
        #    self.nonln.append(torch.nn.LeakyReLU())
        #self.nonln = torch.nn.
        ############################################################
        # Final Dense Layer
        ############################################################
        self.denseout = torch.nn.Linear(len(brchsize),96)#106)

    def forward(self,X):
        brchindex = list(np.asarray([0,50,38,91,8,82,20,20]).cumsum())
        X_u, X_v, X_w, X_th = X[:,brchindex[0]:brchindex[1]],X[:,brchindex[1]:brchindex[2]],X[:,brchindex[2]:brchindex[3]],X[:,brchindex[3]:brchindex[4]]
        X_hdia, X_lw, X_sw = X[:,brchindex[4]:brchindex[5]],X[:,brchindex[5]:brchindex[6]],X[:,brchindex[6]:brchindex[7]]
        ############################################################
        # Optimal PC layer
        ############################################################
        bestv = self.dense1(X_v)
        bestlw = self.dense2(X_lw)
        bestsw = self.dense3(X_sw)
        ############################################################
        # Concat
        ############################################################
        bestPC = torch.cat((bestv,bestlw,bestsw),1)
        bestPC_proc = bestPC
        ############################################################
        # Nonlinear layer
        ############################################################
        for nonln in self.nonln:
            bestPC_proc = nonln(bestPC_proc)
        ############################################################
        # Prediction layer
        ############################################################
        outpred = self.denseout(bestPC_proc)
        return outpred

class OptimMLR_all_3D_lwswhdia(torch.nn.Module):
    def __init__(self,num_nonlinear):
        #super(OptimMLR_all_2D, self).__init__()
        super(OptimMLR_all_3D_lwswhdia, self).__init__()
        ############################################################
        # Input channels
        ############################################################
        brchsize = [82,20,20]
        self.dense1 = torch.nn.Linear(brchsize[0], 1)
        self.dense2 = torch.nn.Linear(brchsize[1], 1)
        self.dense3 = torch.nn.Linear(brchsize[2], 1)
        ############################################################
        # Nonlinear layers
        ############################################################
        self.nonln = torch.nn.ModuleList([
            torch.nn.Sequential(torch.nn.Linear(len(brchsize),len(brchsize)),
                                torch.nn.LeakyReLU()) for i in range(num_nonlinear)])
                                
        #for i in range(num_nonlinear):
        #    self.nonln.append(torch.nn.Linear(7,7))
        #    self.nonln.append(torch.nn.LeakyReLU())
        #self.nonln = torch.nn.
        ############################################################
        # Final Dense Layer
        ############################################################
        self.denseout = torch.nn.Linear(len(brchsize),96)#106)
        
    def forward(self,X):
        brchindex = list(np.asarray([0,50,38,91,8,82,20,20]).cumsum())
        X_u, X_v, X_w, X_th = X[:,brchindex[0]:brchindex[1]],X[:,brchindex[1]:brchindex[2]],X[:,brchindex[2]:brchindex[3]],X[:,brchindex[3]:brchindex[4]]
        X_hdia, X_lw, X_sw = X[:,brchindex[4]:brchindex[5]],X[:,brchindex[5]:brchindex[6]],X[:,brchindex[6]:brchindex[7]]
        ############################################################
        # Optimal PC layer
        ############################################################
        besthdia = self.dense1(X_hdia)
        bestlw = self.dense2(X_lw)
        bestsw = self.dense3(X_sw)
        ############################################################
        # Concat
        ############################################################
        bestPC = torch.cat((besthdia,bestlw,bestsw),1)
        bestPC_proc = bestPC
        ############################################################
        # Nonlinear layer
        ############################################################
        for nonln in self.nonln:
            bestPC_proc = nonln(bestPC_proc)
        ############################################################
        # Prediction layer
        ############################################################
        outpred = self.denseout(bestPC_proc)
        return outpred
    
class R2Loss(torch.nn.Module):
    
    def forward(self, y_pred, y):
        var_y = torch.var(y, unbiased=False)
        return -(1.0 - torch.nn.functional.mse_loss(y_pred, y, reduction="mean") / var_y)
    
class MSE_physicalLoss(torch.nn.Module):
    def __init__(self, pcs=None, wantcomps=None, variances=None):
        super(MSE_physicalLoss, self).__init__()
        self.pcs = pcs
        self.wantcomps = wantcomps
        self.variances = variances
        
    def forward(self, y_pred, y):
        def myscore(y_pred, y, var_y):
            return torch.mean(torch.square(y - y_pred).sum()).div(var_y)
            #var_y = torch.var(y, unbiased=False)
            #return (torch.nn.functional.mse_loss(y_pred, y, reduction="mean") / var_y) #mse/var
        """
        1. Here we slice the feature axis to separate samples into u/v/w/theta components
        """
        y_pred_u, y_u = y_pred[:,0:self.wantcomps[0]], y[:,0:self.wantcomps[0]]
        y_pred_v, y_v = y_pred[:,self.wantcomps[0]:self.wantcomps[0]+self.wantcomps[1]], y[:,self.wantcomps[0]:self.wantcomps[0]+self.wantcomps[1]]
        y_pred_w, y_w = y_pred[:,self.wantcomps[0]+self.wantcomps[1]:self.wantcomps[0]+self.wantcomps[1]+self.wantcomps[2]], \
        y[:,self.wantcomps[0]+self.wantcomps[1]:self.wantcomps[0]+self.wantcomps[1]+self.wantcomps[2]]
        y_pred_th, y_th = y_pred[:,self.wantcomps[0]+self.wantcomps[1]+self.wantcomps[2]:self.wantcomps[0]+self.wantcomps[1]+self.wantcomps[2]+self.wantcomps[3]], \
        y[:,self.wantcomps[0]+self.wantcomps[1]+self.wantcomps[2]:self.wantcomps[0]+self.wantcomps[1]+self.wantcomps[2]+self.wantcomps[3]]
        """
        2. Here we convert PCs into physical space
        """
        uphys_pred, uphys_real = torch.matmul(y_pred_u, self.pcs[0]),torch.matmul(y_u, self.pcs[0])
        vphys_pred, vphys_real = torch.matmul(y_pred_v, self.pcs[1]),torch.matmul(y_v, self.pcs[1])
        wphys_pred, wphys_real = torch.matmul(y_pred_w, self.pcs[2]),torch.matmul(y_w, self.pcs[2])
        thphys_pred, thphys_real = torch.matmul(y_pred_th, self.pcs[3]),torch.matmul(y_th, self.pcs[3])
        
        u_mse, v_mse, w_mse, th_mse = myscore(uphys_pred,uphys_real,self.variances[0]),myscore(vphys_pred,vphys_real,self.variances[1]),\
        myscore(wphys_pred,wphys_real,self.variances[2]),myscore(thphys_pred,thphys_real,self.variances[3])
        
        msesum = u_mse+v_mse+w_mse+th_mse
        return msesum/4#, u_mse,v_mse,w_mse,th_mse
    
def eval_model(model, dataloader, loss_func, metric_func):
    with torch.no_grad():
        loss = 0
        metric = 0
        
        global_sum = 0
        label_size = 0
        for feature, labels in dataloader:
            global_sum += labels.sum()
            label_size += len(labels)
            
        global_mean = global_sum / label_size
        for features, labels in dataloader:
            pred = model(features)
            batch_loss = loss_func(pred, labels)
            batch_metric = metric_func(pred, labels, global_mean)
            
            loss+=batch_loss.item()
            metric+=batch_metric.item()
            
        num_batches = len(dataloader)
        
        loss = loss/num_batches
        metric = metric/num_batches
        return (loss, metric)
    
# Customzied evaluation metric NSE for validation set and test set # 
def calc_nse(sim: torch.FloatTensor, obs: torch.FloatTensor, global_obs_mean: torch.FloatTensor) -> float:
    """Calculate the Nash-Sutcliff-Efficiency coefficient.

    :param obs: Array containing the observations
    :param sim: Array containing the simulations
    :param global_obs_mean: mean of the whole observation series
    :return: NSE value.
    """
    numerator = torch.square(sim - obs).sum()
    #denominator = torch.square(obs - global_obs_mean).sum()
    #nse_val = 1 - numerator / denominator

    return numerator

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            #self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
        
def train_model(model=None,optimizer=None,scheduler=None,numepochs=None,early_stopper=None,variance_store=None,lossfunc=None,regularization='None',l1_lambda=0.01,train_loader=None,val_loader=None,test_loader=None):
    # Custom loss: MSE_physicalLoss(eigenvectors,wcomps,variance_store)
    liveloss = PlotLosses()
    schedulerCY,schedulerLS = scheduler[1],scheduler[0]
    train_losses,trainu_losses,trainv_losses,trainw_losses,trainth_losses = [],[],[],[],[]
    val_losses = []
    val_NSEs = []
    for epoch in (range(int(numepochs))):
        """
        Initialize loss
        """
        train_loss = 0
        """
        Operate per batch
        """
        for features, labels in train_loader:
            optimizer.zero_grad()
            
            prediction = model(features)
            batch_loss = lossfunc(prediction, labels)#loss_func(prediction, labels)
            
            if regularization=='L1':
                #L1 regularization-------------------------------
                l1_norm = sum(abs(p) for p in model.parameters())
                #-------------------------------------------------
                batch_loss = batch_loss + l1_lambda * l1_norm
                batch_loss.backward()
            elif regularization=='None':
                batch_loss.backward()                
            
            optimizer.step()
            schedulerCY.step()
            
            train_loss += batch_loss.item()
            
            
        train_loss = train_loss / len(train_loader)
        train_losses.append(train_loss)
        val_loss, val_NSE = eval_model(model,
                                       val_loader,
                                       lossfunc,
                                       calc_nse)
        schedulerLS.step(val_loss)
        
        ##################################################################
        # Early Stopping (valid / train)
        ##################################################################
        counter = 0
        if len(val_losses)>=1:
            best_score = val_losses[-1]
            if val_loss > best_score:
                counter += 1
                val_NSEs.append(val_NSE)
                val_losses.append(val_loss)
                if counter >= 10:
                    break
            else:
                val_NSEs.append(val_NSE)
                val_losses.append(val_loss)
        else:
            val_NSEs.append(val_NSE)
            val_losses.append(val_loss)
        
        if early_stopper.__call__(val_loss, model):
            break
        #if early_stopper.early_stop(val_loss):             
        #    break
            ##################################################################
        #val_NSEs.append(val_NSE)
        #val_losses.append(val_loss)
        
        if epoch % 500 == 0:
            print((np.log10(train_loss),np.log10(val_loss)))
            
    #return model, {'train':train_losses,'utrain':trainu_losses,'vtrain':trainv_losses,'wtrain':trainw_losses,'thtrain':trainth_losses,'val':val_losses} 
    return model, {'train':train_losses,'val':val_losses} 
