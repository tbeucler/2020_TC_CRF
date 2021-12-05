import numpy as np

##############################################################################################################################################
# Preprocessing fields
def preproc_fields(var=None,timezoom=None,smooth='Yes',gaussian=0.9,fromcenter='Yes',inradius=None,outradius=r500,dostandard='Yes'):
		"""
		Var: Input variable (must have 4 dimensions! Time-pres-theta-radius)
		"""
		if smooth=='Yes':
				from scipy.ndimage import gaussian_filter
				normal_var = []
				if dostandard=='Yes':
						for presindex in range(len(var[0,:,0,0])):
								normal_var.append(gaussian_filter(normalize_inner(var[:,presindex,:,:],outradius,'Yes'),sigma=gaussian))
				elif dostandard=='No':		
						for presindex in range(len(var[0,:,0,0])):		
								normal_var.append(gaussian_filter(normalize_inner(var[:,presindex,:,:],outradius,'No'),sigma=gaussian))            
								normal_var = np.swapaxes(np.asarray(normal_var),0,1)		
		else:
				normal_var = []						
				for presindex in range(len(var[0,:,0,0])):
						normal_var.append(normalize_inner(var[:,presindex,:,:],outradius))
						normal_var = np.swapaxes(np.asarray(normal_var),0,1)
		if fromcenter=='Yes':										
				normal_varf = np.asarray([normal_var[i,:,:,:outradius].flatten() for i in range(len(normal_var[timezoom[0]:timezoom[1],0,0,0]))])														
		elif fromcenter=='No':
				normal_varf = np.asarray([normal_var[i,:,:,inradius:outradius].flatten() for i in range(len(normal_var[timezoom[0]:timezoom[1],0,0,0]))])        
		print("--Finish preprocessing--")
		return normal_var,normal_varf

def do_PCA(var=None,timezoom=None,smooth='Yes',gaussian=0.9,fromcenter='Yes',inradius=None,outradius=None,donormal='Yes',do_PCA='Yes',do_center='No'):
		"""
		var: (4d dimension, time-z-r-theta-r)
		"""
		if smooth=='Yes':
				from scipy.ndimage import gaussian_filter
				normal_var = []
				if donormal=='Yes':
						for presindex in range(len(var[0,:,0,0])):
								normal_var.append(gaussian_filter(normalize_inner(var[:,presindex,:,:],outradius,'Yes'),sigma=gaussian))
				elif donormal=='No':
						if do_center=='Yes':	
								for presindex in range(len(var[0,:,0,0])):
										normal_var.append(gaussian_filter(normalize_inner(var[:,presindex,:,:],outradius,'No','Yes'),sigma=gaussian))
						elif do_center=='No':
								for presindex in range(len(var[0,:,0,0])):
										normal_var.append(gaussian_filter(normalize_inner(var[:,presindex,:,:],outradius,'No','No'),sigma=gaussian))  
						normal_var = np.swapaxes(np.asarray(normal_var),0,1)
		else:
				normal_var = []
				for presindex in range(len(var[0,:,0,0])):
						normal_var.append(normalize_inner(var[:,presindex,:,:],outradius))
				normal_var = np.swapaxes(np.asarray(normal_var),0,1)
		if fromcenter=='Yes':
				normal_varf = np.asarray([normal_var[i,:,:,:outradius].flatten() for i in range(len(normal_var[timezoom[0]:timezoom[1],0,0,0]))])
		elif fromcenter=='No':
				normal_varf = np.asarray([normal_var[i,:,:,inradius:outradius].flatten() for i in range(len(normal_var[timezoom[0]:timezoom[1],0,0,0]))])     
		print("--Finish preprocesing--")
		
		if do_PCA=='Yes':
				from sklearn.decomposition import PCA
				import time
				start_time = time.time()
				skpcaVAR = PCA()
				skpcaVAR.fit(normal_varf.copy())
				print("--- %s seconds ---" % (time.time() - start_time))
				return skpcaVAR,normal_var,normal_varf
		else:
				return normal_var,normal_varf

				
def normalize_inner(var=None,outerradius=None,standard='Yes',docenter='No'):
		PWper_ctrl = []
		for indx in range(len(var[:,0,0])):
				if docenter=='Yes':
						temp = var[indx,:,:outerradius]-np.nanmean(var[indx,:,:outerradius],axis=(0,1))
				elif docenter=='No':
						temp = var[indx,:,:outerradius]
				if standard=='Yes':
						PWper_ctrl.append((temp-np.nanmean(temp))/np.nanstd(temp))
				elif standard=='No':
						PWper_ctrl.append((temp))
		del temp
		return np.asarray(PWper_ctrl)

##############################################################################################################################################
# Reconstruct original field
def reconstruct_fromPCA(PCAdict=None,Afdict=None,Adict=None,component=5,ALL='No'):
		"""
		PCAdict: Processed PCA objects
		Afdict: Flattened processed variable array
		Adict: Processed variable array
		component: Number of component used to reconstruct original array
		ALL: Use all components?
		"""
		if ALL=='Yes':
				Xhat = np.dot(PCAdict.transform(Afdict)[:,:],PCAdict.components_[:,:])	
		else:
				Xhat = np.dot(PCAdict.transform(Afdict)[:,:int(component)],PCAdict.components_[:int(component),:])
				Xhatc = Xhat.copy()
				Xhatc+=np.mean(Afdict, axis=0)		
				TESTrecon = Xhatc.reshape((Xhatc.shape[0],Adict[0].shape[0],Adict[0].shape[1],Adict[0].shape[2]))
		return TESTrecon
