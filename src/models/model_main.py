# import packages
import os, sys
import logging
import warnings
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
		# retrieving optimization to perform
		with FileLock(os.path.join(self.data_path, 'assets_DB.csv.lock')):
			df_assets = pd.read_csv(os.path.join(self.data_path, 'assets_DB.csv'), index_col=0)
			self.to_process = df_assets.loc[df_assets['optimization'] < max_iter].iloc[0]
			init_db = df_assets.loc[self.to_process.name, 'optimization']
			df_assets.loc[self.to_process.name, 'optimization'] += 1
			df_assets.to_csv(os.path.join(self.data_path, 'assets_DB.csv'))
		
		# preparing bayesian optimization
		optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
		STUDY_NAME = f'{self.to_process["ISIN"]}_{self.to_process["data"]}_{self.to_process["model"]}_optuna_study'
		DB_PATH = os.path.join(self.path_model, f'{self.to_process["ISIN"]}_{self.to_process["data"]}_{self.to_process["model"]}_optimization.log')
		storage = optuna.storages.JournalStorage(
			optuna.storages.journal.JournalFileBackend(DB_PATH),
		)
		print(self.to_process)
		print(init_db)
		while True:
			try:
				print('Loading study')
				study = optuna.load_study(storage=storage, study_name=STUDY_NAME)
				break
				
			except:
				if init_db == 0:
					print('Creatind DB')
					study = optuna.create_study(storage=storage, study_name=STUDY_NAME, direction='minimize')
				
				else:
					print('Waiting for DB creation')
					time.sleep(60)
		
		print('Starting BA')
		
		def objective(trial):
			dict_params = self.ANN_model.params_opti(trial, self.to_process['model'])
			
			# preparing datasets
			self.load_data()
			X_data, Y_data, _, _, _ = self.prepro.preprocessing(self.df_features, data_type = self.to_process['data'])
			X_data, Y_data = X_data[:int(0.8*len(X_data))], Y_data[:int(0.8*len(Y_data))]
			
			k = 5
			kf = KFold(n_splits=k, shuffle=True, random_state=42)
			score = []
			
			
			for fold, (train_idx, val_idx) in enumerate(kf.split(X_data)):
				X_train, Y_train = X_data[train_idx], Y_data[train_idx]
				X_val, Y_val = X_data[val_idx], Y_data[val_idx]
			
				dataset = tf.data.Dataset.from_tensor_slices((X_train, {"pred": Y_train})).shuffle(100).batch(dict_params['batch_size'])
				eval_dataset = tf.data.Dataset.from_tensor_slices((X_val, {"pred": Y_val})).batch(dict_params['batch_size'])
								
				model = ANNmodel().model_build(input_shape = X_train.shape[1:], mod_type = self.to_process['model'], **dict_params)
				
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

		study.optimize(objective, n_trials=1, timeout=17000)
		print(study.best_trial)
		
		
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