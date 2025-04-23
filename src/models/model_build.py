#packages
import os, sys
import zipfile
import pandas as pd
import numpy as np
import argparse
import optuna
from fastparquet import write
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, LSTM, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import MultiHeadAttention, Dropout, Add, Reshape, Lambda
from tensorflow.keras.optimizers import Adam

tf.random.set_seed(42)


class ANN_model():
	"""
	Build and fitting model.

	Args:
		job_id (int, optionnal): Slurm job ID.
	"""
	def __init__(self, job_id: int = 0, resampling_unit: str = 'min'):
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
		self.job_id = job_id
		self.resampling_unit = resampling_unit
		
	def model_build(self, input_shape, mod_type, filters: int = 32, kernel_size: slice =(3,3), 
					hidden_units: int = 32, num_layers: int = 1, num_layers_Conv: int = 1, hidden_units_LSTM: int = 16, 
					num_layers_LSTM: int = 1, batch_size: int = 32, epochs: int = 100, **kwargs):
		"""
		Building and compiling CNN 2D model.
		
		Args:
			input_shape (tuple): Dimensions de l'entrée (hauteur, largeur, canaux).

		Returns:
			model (tf.keras.Model): Compiled CNN 2D model.
		"""
		if mod_type == 'CNN':
			input_model = Input(shape=input_shape)
			input_model = Reshape((input_shape[0], input_shape[1], 1))(input_model)
			x = Conv2D(filters=filters, kernel_size=kernel_size, activation='relu')(input_model)
			for _ in range(num_layers_Conv - 1):
				if input_shape[1] < 10:
					s = (3,1)
				else:
					s = (3,3)
				x = Conv2D(filters=filters//2, kernel_size=s, activation='relu')(x)
				
			x = Flatten()(x)
			
			for _ in range(num_layers - 1):
				x = Dense(units=hidden_units, activation='relu')(x)
				
			
		if mod_type == 'LSTM':
			input_model = Input(shape=input_shape)
			x = LSTM(units=hidden_units_LSTM, return_sequences=True, dropout=0.2)(input_model)
			for _ in range(num_layers_LSTM - 1):
				x = LSTM(units=hidden_units_LSTM, dropout=0.2)(x)
				
			x = Flatten()(x)
			
			for _ in range(num_layers - 1):
				x = Dense(units=hidden_units, activation='relu')(x)
		
		
		if mod_type == 'CNN_LSTM':
			input_model = Input(shape=input_shape)
			input_model = Reshape((input_shape[0], input_shape[1], 1))(input_model)
			x = Conv2D(filters=filters, kernel_size=kernel_size, activation='relu')(input_model)
			for _ in range(num_layers_Conv - 1):
				if input_shape[1] < 10:
					s = (3,1)
				else:
					s = (3,3)
				x = Conv2D(filters=filters//2, kernel_size=s, activation='relu')(x)
				
			x = LSTM(units=hidden_units_LSTM, return_sequences=True, dropout=0.2)(x)
			for _ in range(num_layers_LSTM - 1):
				x = LSTM(units=hidden_units_LSTM, dropout=0.2)(x)
				
			x = Flatten()(x)
			
			for _ in range(num_layers - 1):
				x = Dense(units=hidden_units, activation='relu')(x)
			
		
		pred = Dense(units=100, activation='linear')(x)
		
		model = Model(inputs=input_model, outputs=pred)
			
		model.compile(optimizer=Adam(learning_rate=1e-4, clipnorm=1.0), loss='mean_squared_error', metrics=['mse','mae',self.sign_accuracy])
					  
		return model

	def sign_accuracy(self, y_true, y_pred):
		# Compare les signes : True si les signes sont identiques
		y_pred = tf.where(tf.math.is_nan(y_pred), tf.zeros_like(y_pred, dtype=tf.float32), y_pred)
		same_sign = tf.equal(tf.sign(y_true), tf.sign(y_pred))
	
		return tf.reduce_mean(tf.cast(same_sign, tf.float32))
	
	def params_opti(self, trial, mod_type):
		"""
		CNN parameters for optimization process
		"""
		if mod_type == 'CNN':
			trial.suggest_categorical('filters', [32, 64])
			trial.suggest_categorical('kernel_size', [(3,3), (5,5), (7,7)])
			trial.suggest_categorical('hidden_units', [32, 64, 128, 256])
			trial.suggest_categorical('num_layers', [1, 2, 3])
			trial.suggest_categorical('num_layers_Conv', [1, 2])
			trial.suggest_categorical('batch_size', [32, 64, 128])
			trial.suggest_categorical('epochs' , [100, 500, 1000])
		
		elif mod_type == 'LSTM':
			trial.suggest_categorical('hidden_units', [32, 64, 128, 256])
			trial.suggest_categorical('num_layers', [1, 2, 3])
			trial.suggest_categorical('hidden_units_LSTM', [16, 32, 64])
			trial.suggest_categorical('num_layers_LSTM', [1, 2])
			trial.suggest_categorical('batch_size', [32, 64, 128])
			trial.suggest_categorical('epochs' , [100, 500, 1000])
			
		elif mod_type == 'CNN_LSTM':
			trial.suggest_categorical('filters', [32, 64])
			trial.suggest_categorical('kernel_size', [(3,3), (5,5), (7,7)])
			trial.suggest_categorical('hidden_units', [32, 64, 128, 256])
			trial.suggest_categorical('num_layers', [1, 2, 3])
			trial.suggest_categorical('num_layers_Conv', [1, 2])
			trial.suggest_categorical('hidden_units_LSTM', [16, 32, 64])
			trial.suggest_categorical('num_layers_LSTM', [1, 2])
			trial.suggest_categorical('batch_size', [32, 64, 128])
			trial.suggest_categorical('epochs' , [100, 500, 1000])
		
		
		return trial.params
		
		
		
		
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
	args = parser.parse_args()
	
	clust = clustering(args.job_id)    

	if args.slurm_array:
		clust.get_files()
		clust.load_asset_characteristics()
		clust.array_process()