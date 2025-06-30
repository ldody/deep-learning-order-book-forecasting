# import packages
import os, sys
import pandas as pd
import numpy as np
from SALib.sample import saltelli
from SALib.analyze import sobol
from SALib.sample import morris as morris_sample
from SALib.analyze import morris
import tensorflow as tf
import logging
import optuna
from fastparquet import write
import warnings
from filelock import FileLock
import traceback

warnings.simplefilter(action='ignore', category=Warning)
warnings.simplefilter(action='ignore', category=FutureWarning)

src_path = os.path.dirname(os.path.abspath(__file__))
while os.path.basename(src_path) != 'src':
	src_path =  os.path.dirname(src_path)
	
sys.path.append(os.path.join(src_path,'utils'))

from base_log import Base, log_execution

sys.path.append(os.path.join(src_path,'models'))

from model_preprocessing import RegressionPreprocess as prepro
from model_build import ANN_model

class GSA(Base):
	"""
	Performing GSA analysis.

	Args:
		None.
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
		self.path_model = os.path.join(self.root_path,'model')
		self.data_path = os.path.join(self.root_path,'data')
		self.processed_path = os.path.join(self.data_path,'processed')
		self.features_path = os.path.join(self.processed_path,'features')
		self.df_assets = pd.read_csv(os.path.join(self.data_path, 'assets_DB.csv'), index_col=0)
		self.results_path = os.path.join(self.root_path,'results')
		self.architecture_path = os.path.join(self.results_path,'model_architecture')
		self.forecast_path = os.path.join(self.results_path,'forecast')
		self.GSA_path = os.path.join(self.results_path,'GSA')
		self.prepro = prepro()
		self.ANN_model = ANN_model()
		self.to_process = None
		
	def retrieve_files(self):
		"""
		Retrieving forecast files.
		"""
		self.ls_forecast = [f.split('.')[0] for f in os.listdir(self.forecast_path) if 'True' not in f]
		self.ls_GSA = [f.split('.')[0] for f in os.listdir(self.GSA_path)]
		print(self.ls_forecast)
		print(self.ls_GSA)
		
	def load_data(self):
		"""
		Loading data.
		"""
		features_filename = f'{self.to_process["ISIN"]}_features.parquet.gzip'
		self.df_features = pd.read_parquet(os.path.join(self.features_path, features_filename))
		X_data, _, _, _, _ = prepro().preprocessing(data=self.df_features, data_type = self.to_process["data"])
		X_test = X_data[int(0.8*len(X_data)):]
		X = X_test[0] 
		
		self.df_features = self.df_features[:3]
		
		if self.to_process["data"] == 'FOB':
			self.df_features = self.df_features.drop(['Open','High','Low','Volume'], axis=1)
		
		return X
		
	def load_optuna_config(self):
		"""
		Loading optuna config.
		"""
		optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
		STUDY_NAME = f'{self.to_process["file"]}_optuna_study'
		DB_PATH = os.path.join(self.path_model, f'{self.to_process["file"]}_optimization.log')
		storage = optuna.storages.JournalStorage(
			optuna.storages.journal.JournalFileBackend(DB_PATH),
		)
		
		dict_params = optuna.load_study(storage=storage, study_name=STUDY_NAME).best_params
		
		return dict_params
		
		
	def LSA_morris(self):
		"""
		Morris LSA.
		"""
		col_name = [c for c in self.df_features.reset_index().columns if (('price' not in c) & ('target' not in c))]
		col_id = [self.df_features.reset_index().columns.get_loc(col) for col in col_name]
		
		problem = {
			'num_vars': len(col_name),
			'names': col_name,
			'bounds': [[-1, 1]] * len(col_name)
		}

		param_values = morris_sample.sample(problem, N=10, num_levels=4)
		
		def run_model(sample, X_samp, r):
			for i, v in zip(col_id, sample):
				X_samp[r,i] = v
			
			X_samp = X_samp.reshape(1, X_samp.shape[0], X_samp.shape[1], 1)
			
			return self.model.predict(X_samp, verbose=0)['pred']
			
		Y = []
		for row in range(self.X.shape[0]):
		#for row in range(10):
			print(row)
			Y.append(np.array([run_model(params, self.X, row) for params in param_values]))
			
		res = pd.DataFrame()

		for r, Y_r in enumerate(Y):
			for t in range(Y_r.shape[2]):
				# Analyse de Morris
				Si = morris.analyze(problem, param_values, Y_r[:,:,t])
				tmp = pd.DataFrame(Si)
				tmp['row'] = r
				tmp['target'] = t
				res = pd.concat([res, tmp])
		
		df_res = res.groupby(['names','row'], as_index=False).mean()
		results_sorted = df_res.sort_values(by='mu_star', ascending=False).reset_index(drop=True)[:50]
		
		return results_sorted
		
	def GSA_sobol(self, results_sorted):
		"""
		Sobol GSA.
		"""
		len_p = len(results_sorted)
		names_p = results_sorted.loc[:,['names','row']].apply(lambda x: '_'.join(x.astype(str)), axis=1).to_list()

		problem = {
			'num_vars': len_p,
			'names': names_p,
			'bounds': [[-1, 1]] * len_p
		}

		param_values = saltelli.sample(problem, 128, calc_second_order=False)
		
		col_id_s = [self.df_features.reset_index().columns.get_loc(col) for col in results_sorted['names']]
		results_sorted['col'] = 0
		results_sorted.loc[:,'col'] = col_id_s
		
		def run_model_s(param, X_samp, data):
			for i, row in data.iterrows():
				X_samp[row['row'], row['col']] = param[i]
			
			X_samp = X_samp.reshape(1, X_samp.shape[0], X_samp.shape[1], 1)
			
			return self.model.predict(X_samp, verbose=0)['pred']

		Y = np.array([run_model_s(params, self.X, results_sorted[:50]) for params in param_values])
		
		res = pd.DataFrame()

		for t in range(Y.shape[2]):
		#for t in range(4):
			print(t)
			Si = sobol.analyze(problem, Y[:,:,t].reshape(Y.shape[0]), calc_second_order=False)

			tmp = pd.concat(Si.to_df(), axis=1)
			tmp['target'] = t
			res = pd.concat([res,tmp])
		
		return res
	
	def main(self):
		"""
		Executing GSA script.
		"""
		self.retrieve_files()
		
		with FileLock(os.path.join(self.data_path, 'assets_GSA.csv.lock')):
			df_assets = pd.read_csv(os.path.join(self.data_path, 'assets_GSA.csv'), index_col=0)
			df_assets = df_assets[df_assets['data'] != 'OHLCV']
			self.to_process = df_assets.loc[((df_assets['file'].isin(self.ls_forecast)) 
												& (~df_assets['file'].isin(self.ls_GSA))
												& (df_assets['GSA'] == 'PENDING'))].iloc[0]
			
			df_assets.loc[self.to_process.name, 'GSA'] = 'RUNNING'
			df_assets.to_csv(os.path.join(self.data_path, 'assets_GSA.csv'))
			
		print(self.to_process)
		try:
			self.X = self.load_data()
			
			dict_params = self.load_optuna_config()
			
			self.model = self.ANN_model.model_build(input_shape = self.X.shape, mod_type = self.to_process['model'], **dict_params)
			self.model.summary()
			self.model.load_weights(os.path.join(self.architecture_path, f"{self.to_process['file']}.h5"))
			
			res_morris = self.LSA_morris()
			write(os.path.join(self.GSA_path, f"{self.to_process['file']}_LSA.parquet.gzip"), res_morris, compression='GZIP', append=False)
			res_sobol = self.GSA_sobol(res_morris)
			
			write(os.path.join(self.GSA_path, f"{self.to_process['file']}.parquet.gzip"), res_sobol, compression='GZIP', append=False)
			
		except:
			with FileLock(os.path.join(self.data_path, 'assets_GSA.csv.lock')):
				df_assets = pd.read_csv(os.path.join(self.data_path, 'assets_GSA.csv'), index_col=0)
				
				df_assets.loc[self.to_process.name, 'GSA'] = 'PENDING'
				df_assets.to_csv(os.path.join(self.data_path, 'assets_GSA.csv'))
				traceback.print_exc()
		

if __name__ == "__main__":
	sens_ana = GSA()
	sens_ana.main()