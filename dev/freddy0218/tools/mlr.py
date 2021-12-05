import numpy as np
from sklearn.linear_model import LinearRegression
#from sklearn.linear_model import MultiTaskLassoCV
from celer import MultiTaskLassoCV
from celer import Lasso
from sklearn.linear_model import ElasticNet

def produce_regscore(inPUT=None,outPUT=None,aux_inPUT=None,outtype='score',do_aux=False,algorithm='linear',lassoparam=dict(tol=1e-6,cv=4,n_jobs=1,n_alphas=20),simplassoalpha=0.1):
		if algorithm=='linear':
				reg = LinearRegression().fit(inPUT, outPUT)
		elif algorithm=='lasso':
				#reg = MultiTaskLassoCV(**lassoparam).fit(inPUT,outPUT)
				reg = Lasso(simplassoalpha).fit(inPUT,outPUT)					
		elif algorithm=='multilasso':											
				reg = MultiTaskLassoCV(**lassoparam).fit(inPUT,outPUT)

		# Output
		#................................................................
		if outtype=='score':
				if do_aux is False:
						return reg.score(inPUT, outPUT)
				elif do_aux is True:
						return reg.score(aux_inPUT,outPUT)
		elif outtype=='predict':
				return reg.predict(inPUT)
		elif outtype=='coeff':
				return reg.coef_

def delete_padding(inTS=None,outTS=None):
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

def output_regscore(inTS=None,outTS=None,LTlist=None,algorithm='linear',aux_inTS=None,do_aux=False):
		if do_aux is False:
				return [produce_regscore(delete_padding(inTS,outTS[i])[0],delete_padding(inTS,outTS[i])[1],None,'score',False,algorithm,dict(tol=1e-6,cv=4,n_jobs=1,n_alphas=20),0.01) for i in range(len(outTS))]
		elif do_aux is True:
				return [produce_regscore(delete_padding(inTS,outTS[i])[0],delete_padding(inTS,outTS[i])[1],aux_inTS[:-int(LTlist[i])],'score',True,algorithm,dict(tol=1e-6,cv=4,n_jobs=1,n_alphas=20),0.01) for i in range(len(outTS))]

def output_regscore_withmemory(inTS=None,outTS=None,LTlist=None,algorithm='linear',aux_inTS=None,do_aux=False):
		if do_aux is False:
				output = []
				for i in (range(len(outTS))):
						inTSn,outTSn = delete_padding(inTS,outTS[i])[0],delete_padding(inTS,outTS[i])[1]
						inTRAIN = np.concatenate(((inTSn[int(LTlist[i]):]),(inTSn[0:-int(LTlist[i])])),axis=1)
						outTRAIN = outTSn[int(LTlist[i]):]
						output.append(produce_regscore(inTRAIN,outTRAIN,None,'score',False,algorithm,dict(tol=1e-6,cv=4,n_jobs=1,n_alphas=20),0.01))
				return output
		elif do_aux is True:
				output = []
				for i in (range(len(outTS))):
						inTSn,outTSn = delete_padding(inTS,outTS[i])[0],delete_padding(inTS,outTS[i])[1]
						inTRAIN = np.concatenate(((inTSn[int(LTlist[i]):]),(inTSn[0:-int(LTlist[i])])),axis=1)
						outTRAIN = outTSn[int(LTlist[i])]
						auxin = np.concatenate(((aux_inTS[:-int(LTlist[i])][int(LTlist[i]):]),(aux_inTS[:-int(LTlist[i])][0:-int(LTlist[i])])),axis=1)
						output.append(produce_regscore(inTRAIN,outTRAIN,auxin,'score',True,algorithm,dict(tol=1e-6,cv=4,n_jobs=1,n_alphas=20),0.01))
				return output        
																																															    
def output_regcoeff(inTS=None,outTS=None,LTlist=None,algorithm='linear',memory='with'):
		if memory=='with':
				output = []
				for i in range(len(outTS)):
						inarray = np.concatenate(((inTS[:-int(LTlist[i])][int(LTlist[i]):]),(inTS[:-int(LTlist[i])][0:-int(LTlist[i])])),axis=1)
						outarray = outTS[i][int(LTlist[i]):-int(LTlist[i])]
						output.append(produce_regscore(inarray,outarray,'coeff',algorithm,dict(tol=1e-6,cv=4,n_jobs=1,n_alphas=20),0.01))
				return output
		elif memory=='without':
				return [produce_regscore(inTS[:-int(LTlist[i])],outTS[i][:-int(LTlist[i])],'coeff',algorithm,dict(tol=1e-6,cv=4,n_jobs=1,n_alphas=20),0.01) for i in range(len(outTS))]

def output_regcoeff(inTS=None,outTS=None,LTlist=None,algorithm='linear',memory='with'):
		if memory=='with':
				output = []
				for i in range(len(outTS)):
						inarray = np.concatenate(((inTS[:-int(LTlist[i])][int(LTlist[i]):]),(inTS[:-int(LTlist[i])][0:-int(LTlist[i])])),axis=1)
						outarray = outTS[i][int(LTlist[i]):-int(LTlist[i])]
						output.append(produce_regscore(inarray,outarray,'coeff',algorithm,dict(tol=1e-6,cv=4,n_jobs=1,n_alphas=20),0.01))
				return output		
		elif memory=='without':
				return [produce_regscore(inTS[:-int(LTlist[i])],outTS[i][:-int(LTlist[i])],'coeff',algorithm,dict(tol=1e-6,cv=4,n_jobs=1,n_alphas=20),0.01) for i in range(len(outTS))]

def recon_from_linear(forecast_eiginput=None,PCA_dict=None,LT=None,numcomp=[11,12,12],large_out=False):
		def output_reshapeRECON(forecast_eig=None,PCAdict=None):
				testrec_dudt = np.dot(forecast_eig[:,0:numcomp[0]],(PCA_dict['u'].components_[0:numcomp[0]]))#.reshape((91,39,360,167))
				testrec_dvdt = np.dot(forecast_eig[:,numcomp[0]:numcomp[0]+numcomp[1]],(PCA_dict['v'].components_[0:numcomp[1]]))#.reshape((91,39,360,167))
				testrec_dwdt = np.dot(forecast_eig[:,numcomp[0]+numcomp[1]:],(PCA_dict['w'].components_[0:numcomp[2]]))#.reshape((39,360,167))
				return testrec_dudt,testrec_dvdt,testrec_dwdt
		######################################################################################################################################################
		name = ['dudt','dvdt','dwdt']
		temp1,temp2,temp3 = [],[],[]
		temp1b,temp2b,temp3b = [],[],[]
		for i in tqdm(range(len(LT))):
				teMP1,teMP2,teMP3 = output_reshapeRECON(forecast_eiginput[i],PCA_dict)
				reteMP1,reteMP2,reteMP3 = windrates_real(LT=int(i+1))
				# Square error
				if large_out is False:
						temp1.append(np.sum((teMP1-reteMP1)**2))
				else:
						temp1.append(((teMP1-reteMP1)))
				del reteMP1
				gc.collect()

				if large_out is False:
						temp2.append(np.sum((teMP2-reteMP2)**2))
				else:
						temp2.append(((teMP2-reteMP2)))
				del reteMP2
				gc.collect()

				if large_out is False:
						temp3.append(np.sum((teMP3-reteMP3)**2))
				else:
						temp3.append(((teMP3-reteMP3)))
				del reteMP3
				gc.collect()


				# Variance
				if large_out is False:
						temp1b.append(np.sum((teMP1-np.nanmean(teMP1))**2))
				else:
						temp1b.append(((teMP1-np.nanmean(teMP1))))
				del teMP1
				gc.collect()
				if large_out is False:
						temp2b.append(np.sum((teMP2-np.nanmean(teMP2))**2))
				else:
						temp2b.append((teMP2-np.nanmean(teMP2)))
				del teMP2
				gc.collect()
				if large_out is False:
						temp3b.append(np.sum((teMP3-np.nanmean(teMP3))**2))
				else:
						temp3b.append(((teMP3-np.nanmean(teMP3))))
				del teMP3
				gc.collect()
				del i
				se_store = {name[0]:temp1,name[1]:temp2,name[2]:temp3}
				va_store = {name[0]:temp1b,name[1]:temp2b,name[2]:temp3b}
				######################################################################################################################################################
				return se_store,va_store
