import xarray as xr
import numpy as np

def read_some_azimuth_fields(fileloc=None,fieldname=None):
		dict_name = {}
		for inx,obj in enumerate(fileloc):
				field_read = xr.open_dataset(obj)
				dict_name[fieldname[inx]] = field_read
		return dict_name

def add_ctrl_before_senstart(CTRLvar=None,SENvar=None,exp='NCRF36',firstdo='Yes'):
		if firstdo=='Yes':
				if exp=='NCRF36':
						return np.concatenate((CTRLvar[0:36],SENvar))
				elif exp=='NCRF60':
						return np.concatenate((CTRLvar[0:60],SENvar))
		else:
				return SENvar
