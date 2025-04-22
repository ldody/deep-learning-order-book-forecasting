#packages
import os, sys
import zipfile
import pandas as pd
import numpy as np
import argparse
from fastparquet import write
from sklearn.preprocessing import MinMaxScaler

src_path = os.path.dirname(os.path.abspath(__file__))
while os.path.basename(src_path) != 'src':
	src_path =  os.path.dirname(src_path)
	
sys.path.append(os.path.join(src_path,'utils'))

from base_log import Base, log_execution
from NormStand import TanhNormalizer


class RegressionPreprocess(Base):
	"""
	Preprocessing FOB to create LOB.

	Args:
		job_id (int, optionnal): Slurm job ID.
	"""
	def __init__(self, job_id: int = 0, var: float = 1.0, n_pred: int = 100, look_back: int = 100):
		"""
		Initializes the FOBPreprocessor instance.
		
		Attributes:
			path (str): Path of the current script.
			root_path (str): Root path of the project.
			job_id (int): Slurm job ID.
			raw_path (str): Path of the repository with raw data of FOB /data/raw/FOB/.
			processed_path (str): Path of the repository with processed data of FOB /data/processed/FOB/.
			processed_path (str): Path of the repository with processed data of LOB /data/processed/FOB/LOB/.
			processed_path (str): Path of the repository with processed data of FO /data/processed/FOB/FO/.
			fobdm (Class): Class from the FOB database management.
			FOB (DataFrame): FOB DataFrame.
			LOB (DataFrame): LOB DataFrame.
			FO (DataFrame): FO DataFrame.
			filename_tmp (str): Name of the temporary parquet file with LOB dataframe.
			filename_zip (str): Name of the gzip file with the final parquet file with LOB/FO dataframe.
			file (str): Name of the FOB file in process.
			isin (str): Name of the ISIN in process.
			resampling_unit (str): Rule of resampling for the FOB.
			
		Args:
			job_id (int, optionnal): Slurm job ID, Default=0.
		"""
		self.path = os.path.dirname(os.path.abspath(__file__))
		self.root_path = self.path
		while '.venv' not in os.listdir(self.root_path):
			self.root_path =  os.path.dirname(self.root_path)
			
		self.n_pred = n_pred
		self.look_back = look_back
		self.var = var
		
	def preprocessing(self, data, data_type : str = 'OHLCV'):
		'''
		Preprocessing FOB for models.
		'''
		data = self.create_target(data)
		
		data, scaler_p, scaler_v, scaler_t = self.scaling(data)
		
		col = ['index','Open','High','Low']
		
		if data_type == 'OHLCV':
			col = ['index'] + ['Open','High','Low','Close','Volume'] + [c for c in data.columns if 'target' in c]
			data = data[col]
		
		elif data_type == 'FOB':
			data = data.drop(columns=['Open','High','Low','Volume'])
		
		X, Y = self.prepare_sequences(data)
		
		return X, Y, scaler_p, scaler_v, scaler_t
		
	def create_target(self, data):
		'''
		Creating Y True.
		'''
		for i in range(self.n_pred):
			data[f'target_{i+1}'] = data['Close'].shift(-i)
			data[f'target_{i+1}'] = data[f'target_{i+1}']/data['Close'] - 1
		
		data = data.dropna()
		return data

	def prepare_sequences(self, data):
		'''
		Preparing the sequences.
		'''
		data = data.to_numpy()
		X, y = [], []
		for i in range(len(data) - self.look_back - self.n_pred-1):
			X.append(data[i:i + self.look_back, :-self.n_pred])
			y.append(data[i + self.look_back, -self.n_pred:])
			
		return np.array(X), np.array(y)
		
	def scaling(self, data):
		'''
		Scaling data.
		'''
		data = data.reset_index()
		
		scaler_p = TanhNormalizer().fit(data['Close'])
		scaler_v = TanhNormalizer().fit(data['Volume'])
		scaler_t = MinMaxScaler()
		data['index'] = scaler_t.fit_transform(data['index'].to_numpy().reshape(-1, 1))
		
		col_p = ['Open','High','Low','Close'] + [c for c in data.columns if 'price' in c]
		col_v = ['Volume'] + [c for c in data.columns if 'size' in c]
		
		for c in col_p:
			data[c] = scaler_p.transform(data[c])
		
		for c in col_v:
			data[c] = scaler_v.transform(data[c])
			
		return data, scaler_p, scaler_v, scaler_t