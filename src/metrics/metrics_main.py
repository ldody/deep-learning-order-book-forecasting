# import packages
import os, sys
import logging
import warnings
import traceback
import pandas as pd
import numpy as np
import argparse
import time
from fastparquet import write
from datetime import datetime, timedelta
import sklearn
from sklearn.metrics import mean_squared_error, mean_absolute_error

'''import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
tf.config.threading.set_intra_op_parallelism_threads(60)
tf.config.threading.set_inter_op_parallelism_threads(60)
tf.random.set_seed(42)'''

warnings.simplefilter(action='ignore', category=Warning)
warnings.simplefilter(action='ignore', category=FutureWarning)

src_path = os.path.dirname(os.path.abspath(__file__))
while os.path.basename(src_path) != 'src':
	src_path =  os.path.dirname(src_path)
	
sys.path.append(os.path.join(src_path,'utils'))

from base_log import Base, log_execution

class Evaluation(Base):
	"""
	Evaluating models.

	Args:
		job_id (int, optionnal): Slurm job ID.
	"""
	def __init__(self):
		"""
		Initializing the regression instance.
		
		Attributes:

			
		Args:
			job_id (int, optionnal): Slurm job ID, Default=0.
		"""
		self.path = os.path.dirname(os.path.abspath(__file__))
		self.root_path = self.path
		while '.venv' not in os.listdir(self.root_path):
			self.root_path =  os.path.dirname(self.root_path)
		self.data_path = os.path.join(self.root_path,'data')
		self.df_assets = pd.read_csv(os.path.join(self.data_path, 'assets_DB.csv'), index_col=0)
		self.results_path = os.path.join(self.root_path,'results')
		self.forecast_path = os.path.join(self.results_path,'forecast')
		self.metrics_path = os.path.join(self.results_path,'metrics')
		
	def retrieve_files(self):
		"""
		Retrieving forecast files.
		"""
		ls_true = [f for f in os.listdir(self.forecast_path) if 'True' in f]
		ls_pred = [f for f in os.listdir(self.forecast_path) if 'True' not in f]
		
		return ls_true, ls_pred
		
	def calc_metrics(self):
		"""
		Calculating the metrics.
		"""
		ls_true, ls_pred = self.retrieve_files()
		
		
		for file_true in ls_true:
			data_true = pd.read_parquet(os.path.join(self.forecast_path, file_true))
			
			ls_assoc = [f for f in ls_pred if file_true.split('_')[0] == f.split('_')[0]]
			
			for file_pred in ls_assoc:
				data_pred = pd.read_parquet(os.path.join(self.forecast_path, file_pred))
				
				isin = file_pred.split('_')[0]
				data_mod = '_'.join(file_pred.split('.')[0].split('_')[1:])
				print(isin, data_mod, file_true, file_pred)
				print(len(data_pred), len(data_true))
				
				mse = data_true.apply(lambda row: mean_squared_error(row, data_pred.loc[row.name]), axis=1)
				tmp = pd.DataFrame(mse, columns=['mse'])
				tmp[['isin','data_mod']] = isin, data_mod
				tmp['id_forecast'] = tmp.index
				
				sign_match = np.sign(data_true) == np.sign(data_pred)
				accuracy_row = sign_match.mean(axis=1)
				
				mae = data_true.apply(lambda row: mean_absolute_error(row, data_pred.loc[row.name]), axis=1)
				
				tmp['accuracy'] = accuracy_row
				tmp['mae'] = mae
				
				if "metrics_row.parquet.gzip" in os.listdir(self.metrics_path):
					write(os.path.join(self.metrics_path, "metrics_row.parquet.gzip"), tmp, compression='GZIP', append=True)
				else:
					write(os.path.join(self.metrics_path, "metrics_row.parquet.gzip"), tmp, compression='GZIP', append=False)
				
				mse = data_true.apply(lambda col: mean_squared_error(col, data_pred[col.name]), axis=0)
				tmp = pd.DataFrame(mse, columns=['mse'])
				tmp[['isin','data_mod']] = isin, data_mod
				tmp['target'] = tmp.index
				tmp['target'] = tmp['target'].apply(lambda row: int(row.split('_')[1]))
				
				sign_match = np.sign(data_true) == np.sign(data_pred)
				accuracy_row = sign_match.mean(axis=0)
				
				mae = data_true.apply(lambda col: mean_absolute_error(col, data_pred[col.name]), axis=0)
				
				tmp['accuracy'] = accuracy_row
				tmp['mae'] = mae
				
				if "metrics_col.parquet.gzip" in os.listdir(self.metrics_path):
					write(os.path.join(self.metrics_path, "metrics_col.parquet.gzip"), tmp, compression='GZIP', append=True)
				else:
					write(os.path.join(self.metrics_path, "metrics_col.parquet.gzip"), tmp, compression='GZIP', append=False)
					
					
if __name__ == "__main__":
	m = Evaluation()
	m.calc_metrics()