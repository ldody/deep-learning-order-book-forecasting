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
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split, KFold
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
from fastparquet import write
from datetime import datetime, timedelta

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
		self.results_path = os.path.join(self.root_path,'results')
		self.architecture_path = os.path.join(self.results_path,'model_architecture')
		self.forecast_path = os.path.join(self.results_path,'forecast')
		self.training_path = os.path.join(self.results_path,'training')
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
		# function to prepare optuna config
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
			df_assets[['optimization', 'process']] = df_assets[['optimization', 'process']].astype(str)
			
			if len(df_assets.loc[df_assets['process'] != 'True']) == 0:
					sys.exit('All processes performed')
					
			else:
				print(df_assets, df_assets[df_assets['optimization'] == 'True'])
				self.to_process = df_assets.loc[(df_assets['process'] == 'False') & (df_assets['optimization'] == 'True')]
				print(self.to_process)
				self.to_process['joined'] = self.to_process[['ISIN','data','model']].astype(str).apply(lambda x: '_'.join(x), axis=1)
				self.to_process = self.to_process.iloc[0]
				df_assets.loc[self.to_process.name, 'process'] = 'running'
				df_assets.to_csv(os.path.join(self.data_path, 'assets_DB.csv'))
		
		keras_file = f"{self.to_process['joined']}.h5"
		print(keras_file, type(keras_file), os.path.join(self.architecture_path, keras_file))
		
		STUDY_NAME, DB_PATH, storage = load_optuna_config()
		print(STUDY_NAME, DB_PATH)
		#retrieving best combi
		dict_params = optuna.load_study(storage=storage, study_name=STUDY_NAME).best_params
		
		self.load_data()
		X_data, Y_data, _, _, _ = self.prepro.preprocessing(self.df_features, data_type = self.to_process['data'])
		slice_pos = int(0.5*len(Y_data))
		X_, Y_ = X_data[:int(0.8*len(Y_data))], Y_data[:int(0.8*len(Y_data))]
		X_test, Y_test = X_data[int(0.8*len(X_data)):], Y_data[int(0.8*len(Y_data)):]
		
		dataset = tf.data.Dataset.from_tensor_slices((X_, {"pred": Y_})).shuffle(100)
		test_dataset = tf.data.Dataset.from_tensor_slices((X_test, {"pred": Y_test})).batch(dict_params['batch_size'])
			
		if keras_file in os.listdir(self.architecture_path):
			model = load_model(os.path.join(self.architecture_path, keras_file), custom_objects={'sign_accuracy': self.ANN_model.sign_accuracy})
			X_test, Y_test = X_data[int(0.8*len(X_data)):], Y_data[int(0.8*len(Y_data)):]
			
		else:
			print(dataset.take(slice_pos).cardinality().numpy(), dataset.skip(slice_pos).cardinality().numpy(), tf.data.Dataset.from_tensor_slices((X_test, {"pred": Y_test})).cardinality().numpy())
			
			train_dataset = dataset.take(slice_pos).batch(dict_params['batch_size'])
			eval_dataset = dataset.skip(slice_pos).batch(dict_params['batch_size'])
			
			print(f'params {dict_params}')
			print(f'train {train_dataset} \neval {eval_dataset} \ntest {test_dataset}')
			
			#building model
			model = self.ANN_model.model_build(input_shape = X_data.shape[1:], mod_type = self.to_process['model'], **dict_params)

			#fitting model
			checkpoint = ModelCheckpoint(
										filepath=os.path.join(self.architecture_path, keras_file),
										monitor='val_loss',
										save_best_only=True,
										save_weights_only=False,
										mode='min',
										verbose=1
									)

			history = model.fit(train_dataset, 
					  epochs=dict_params['epochs'],  
					  verbose=2,
					  validation_data=eval_dataset,
					  callbacks=[checkpoint]
					  )
			
			pd.DataFrame(history.history).to_csv(os.path.join(self.training_path, f"{self.to_process['joined']}.csv"))
			write(os.path.join(self.training_path, f"{self.to_process['joined']}.parquet.gzip"), pd.DataFrame(history.history), compression='GZIP', append=False)
			
		
		#forecasting
		results = model.predict(test_dataset)
		
		col_names = [f'Target_{i+1}' for i in range(results['pred'].shape[1])]
		
		write(os.path.join(self.forecast_path, f"{self.to_process['joined']}.parquet.gzip"), pd.DataFrame(results['pred'], columns=col_names), compression='GZIP', append=False)
		if f"{self.to_process['ISIN']}.parquet.gzip" not in os.listdir(self.forecast_path):
			write(os.path.join(self.forecast_path, f"{self.to_process['ISIN']}_True.parquet.gzip"), pd.DataFrame(Y_test, columns=col_names), compression='GZIP', append=False)
		
		model.summary()
		
		#modifying DB
		with FileLock(os.path.join(self.data_path, 'assets_DB.csv.lock')):
			df_assets = pd.read_csv(os.path.join(self.data_path, 'assets_DB.csv'), index_col=0)
			df_assets.loc[self.to_process.name, 'process'] = True
			df_assets.to_csv(os.path.join(self.data_path, 'assets_DB.csv'))
		
	def optimization(self, max_iter : int = 25):
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
				if len(df_assets.loc[df_assets['optimization'] != 'True']) == 0:
					sys.exit('All optimizations performed')
				
				else:
					self.to_process = df_assets.loc[df_assets['optimization'] != 'True'][:30].sample(n=1).iloc[0]
					STUDY_NAME, DB_PATH, storage = load_optuna_config()
					print(STUDY_NAME)
				
				try:
					study = optuna.load_study(storage=storage, study_name=STUDY_NAME)
					df = study.trials_dataframe()
					now = datetime.now()
					n = max(len(df.loc[(df['state'] == 'COMPLETE') | ((df['state'] == 'RUNNING') & (df['datetime_start'] > (now - timedelta(days=1))))]), 1)
				except:
					traceback.print_exc()
					study = optuna.create_study(storage=storage, study_name=STUDY_NAME, direction='minimize')
					n = 0
					
				try:
					print(f'number of combinations done: {n}')
				except:
					None
				
				try:
					if len(df) > max_iter*1.2:
						n = max_iter
				except: None
				
				if n >= max_iter:
					df_assets.loc[self.to_process.name, 'optimization'] = True
					df_assets.to_csv(os.path.join(self.data_path, 'assets_DB.csv'))
							
		
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

			try:				
				model = ANN_model().model_build(input_shape = X_data.shape[1:], mod_type = self.to_process['model'], **dict_params)
				for fold, (train_idx, val_idx) in enumerate(kf.split(X_data)):
					X_train, Y_train = X_data[train_idx], Y_data[train_idx]
					X_val, Y_val = X_data[val_idx], Y_data[val_idx]

					dataset = tf.data.Dataset.from_tensor_slices((X_train, {"pred": Y_train})).shuffle(100).batch(dict_params['batch_size'])
					eval_dataset = tf.data.Dataset.from_tensor_slices((X_val, {"pred": Y_val})).batch(dict_params['batch_size'])

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

		try:
			study.optimize(objective, n_trials=1, timeout=17000)
		except:
			with open(os.path.join(self.path_model, '{self.to_process["ISIN"]}_{self.to_process["data"]}_{self.to_process["model"]}.err'), "w") as f:
				traceback.print_exc(file=f)
			sys.exit(1)
		

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