# import packages
import os, sys
import logging
import warnings
import traceback
import pandas as pd
import numpy as np
import argparse
import pickle
from filelock import FileLock
import optuna
import time
from joblib import Parallel, delayed
import multiprocessing
import tensorflow as tf
from sklearn.model_selection import train_test_split, KFold

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
from model_preprocessing import RegressionPreprocess as prepro
from model_build import ANN_model

class Regression(Base):
	"""
	Forecasting prices.

	Args:
		job_id (int, optionnal): Slurm job ID.
	"""
	def __init__(self, job_id: int = 0, resampling_unit: str = 'min'):
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
		self.job_id = job_id
		self.path_model = os.path.join(self.root_path,'model')
		self.data_path = os.path.join(self.root_path,'data')
		self.processed_path = os.path.join(self.data_path,'processed')
		self.features_path = os.path.join(self.processed_path,'features')
		self.df_assets = pd.read_csv(os.path.join(self.data_path, 'assets_DB.csv'), index_col=0)
		self.resampling_unit = resampling_unit
		self.prepro = prepro()
		self.ANN_model = ANN_model()
		self.to_process = None
		
	def load_data(self):
		"""
		Loading data.
		"""
		features_filename = f'{self.to_process["ISIN"]}_features.parquet.gzip'
		self.df_features = pd.read_parquet(os.path.join(self.features_path, features_filename))
		
	def array_process(self):
		"""
		Running script with slurm array jobs
		"""
		self.to_process = self.df_assets.loc[self.job_id]
		self.load_data()
		X_data, Y_data, scaler_p, scaler_v, scaler_t = self.prepro.preprocessing(self.df_features)
		
		model = self.ANN_model.model_build(input_shape = X_data.shape[1:], mod_type='CNN')
		
		print(X_data, Y_data, scaler_p.__dict__, scaler_v.__dict__, scaler_t.__dict__)
		
	def optimization(self, max_iter : int = 200):
		"""
		Running hyperparameters optimization process.
		"""
		# preparing bayesian optimization
		def load_optuna_config():
			optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
			STUDY_NAME = f'{self.to_process["ISIN"]}_{self.to_process["data"]}_{self.to_process["model"]}_optuna_study'
			DB_PATH = os.path.join(self.path_model, f'{self.to_process["ISIN"]}_{self.to_process["data"]}_{self.to_process["model"]}_optimization.log')
			storage = optuna.storages.JournalStorage(
				optuna.storages.journal.JournalFileBackend(DB_PATH),
			)
			return STUDY_NAME, DB_PATH, storage
		
		# retrieving optimization to perform
		with FileLock(os.path.join(self.data_path, 'assets_DB.csv.lock')):
			df_assets = pd.read_csv(os.path.join(self.data_path, 'assets_DB.csv'), index_col=0)
			
			n = max_iter + 1
			while n >= max_iter:
				if len(df_assets.loc[df_assets['optimization'] != True]) == 0:
					sys.exit('All optimizations performed')
				
				else:
					self.to_process = df_assets.loc[df_assets['optimization'] != True].iloc[0]
					STUDY_NAME, DB_PATH, storage = load_optuna_config()
				
				try:
					study = optuna.load_study(storage=storage, study_name=STUDY_NAME)
					df = study.trials_dataframe()
					n = max(len(df.loc[(df['state'] == 'COMPLETE') | (df['state'] == 'RUNNING')]), 1)
				except:
					traceback.print_exc()
					study = optuna.create_study(storage=storage, study_name=STUDY_NAME, direction='minimize')
					n = 0
					
				if n >= max_iter:
					df_assets.loc[self.to_process.name, 'optimization'] = True
					df_assets.to_csv(os.path.join(self.data_path, 'assets_DB.csv'))
							
		
		print('Starting BA')
		
		def objective(trial):
			dict_params = self.ANN_model.params_opti(trial, self.to_process['model'])
			
			# preparing datasets
			self.load_data()
			print(self.features.dtypes)
			X_data, Y_data, _, _, _ = self.prepro.preprocessing(self.df_features, data_type = self.to_process['data'])
			X_data, Y_data = X_data[:int(0.8*len(X_data))], Y_data[:int(0.8*len(Y_data))]
			
			k = 5
			kf = KFold(n_splits=k, shuffle=True, random_state=42)
			score = []

			try:				
				model = ANN_model().model_build(input_shape = X_data.shape[1:], mod_type = self.to_process['model'], **dict_params)
				for fold, (train_idx, val_idx) in enumerate(kf.split(X_data)):
					X_train, Y_train = X_data[train_idx], Y_data[train_idx]
					X_val, Y_val = X_data[val_idx], Y_data[val_idx]

					dataset = tf.data.Dataset.from_tensor_slices((X_train, {"pred": Y_train})).shuffle(100).batch(dict_params['batch_size'])
					eval_dataset = tf.data.Dataset.from_tensor_slices((X_val, {"pred": Y_val})).batch(dict_params['batch_size'])
					print(dataset)
					print('Start fitting model')
					callback = LimitTrainingTime(17000)
					start_time = time.time()
					history = model.fit(dataset, 
										  epochs=dict_params['epochs'],  
										  verbose=2,
										  validation_data=eval_dataset,
										  callbacks=[callback]
										  )
					end_time = time.time()
					
					if end_time - start_time > 17000:
						raise optuna.exceptions.TrialPruned()
						
					else:
						score.append(min(history.history['val_loss'][30:]))
				
				return np.mean(score)
				
			except:
				traceback.print_exc()
				raise optuna.exceptions.TrialPruned()

		study.optimize(objective, n_trials=1, timeout=17000)
		

class LimitTrainingTime(tf.keras.callbacks.Callback):
	def __init__(self, max_time_s):
		super().__init__()
		self.max_time_s = max_time_s
		self.start_time = None

	def on_train_begin(self, logs):
		self.start_time = time.time()

	def on_train_batch_end(self, batch, logs):
		now = time.time()
		if now - self.start_time >  self.max_time_s:
			self.model.stop_training = True

		
#convert str to bool for argparse
def str2bool(v):
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')
	

if __name__ == "__main__":
	#retrieving arguments if any, specify processing way (slurm, parallelism, classic)
	parser = argparse.ArgumentParser()
	parser.add_argument('--job_id', type=int, default=0)
	parser.add_argument('--slurm_array', '-sa', type=str2bool, default=False)
	parser.add_argument('--combined', '-cmb', type=str2bool, default=False)
	parser.add_argument('--bayesian_opti', '-ba', type=str2bool, default=False)
	args = parser.parse_args()
	
	reg = Regression(args.job_id)

	if args.slurm_array:
		reg.array_process()
		
	elif args.combined:
		reg.combined_data_process()
	
	elif args.bayesian_opti:
		print('Launch opti')
		reg.optimization()