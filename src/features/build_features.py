#packages
import os, sys
import zipfile
import pandas as pd
import numpy as np
from fastparquet import write

src_path = os.path.dirname(os.path.abspath(__file__))
while os.path.basename(src_path) != 'src':
	src_path =  os.path.dirname(src_path)
	
sys.path.append(os.path.join(src_path,'utils'))
from base_log import Base, log_execution

class BuildFeatures(Base):
	"""
	Building features data.

	Args:
		job_id (int, optionnal): Slurm job ID.
	"""
	def __init__(self, job_id: int = 0, resampling_unit: str = '5min', **kwargs):
		"""
		Initializing the build features instance.
		
		Attributes:

			
		Args:
			job_id (int, optionnal): Slurm job ID, Default=0.
		"""
		self.path = os.path.dirname(os.path.abspath(__file__))
		self.root_path = self.path
		while all(file not in os.listdir(self.root_path) for file in ['data','results','src']):
			self.root_path =  os.path.dirname(self.root_path)
		self.job_id = job_id
		self.resampling_unit = resampling_unit
		self.data_path = os.path.join(self.root_path,'data')
		self.ohlcv_path = os.path.join(self.data_path,'raw','OHLCV')
		self.processed_path = os.path.join(self.data_path,'processed')
		self.features_path = os.path.join(self.processed_path,'features')
		self.FOB_path_LOB = os.path.join(self.processed_path,'FOB')
		self.df_assets = pd.read_csv(os.path.join(self.data_path, 'assets.csv'), index_col=0)
		self.job_id = job_id
		self.features_df = pd.DataFrame()
		self.to_process = pd.DataFrame()
		
	def check_features(self):
		"""
		Checking existing features
		"""
		ls_isin = [f.split('_')[0] for f in os.listdir(self.features_path)]
		self.df_assets = self.df_assets.loc[~self.df_assets['ISIN'].isin(ls_isin)].reset_index(drop=True)
		if len(self.df_assets) == 0:
			sys.exit('All features already built')
		
		else:
			self.to_process = self.df_assets.loc[self.job_id]
	
	def load_ohlcv(self):
		"""
		Loading OHLCV
		"""
		ohlcv_file = [f for f in os.listdir(self.ohlcv_path) if self.to_process['RIC'] == f.split('.')[0]]
		ohlcv_df = pd.read_csv(os.path.join(self.ohlcv_path, ohlcv_file))
		ohlcv_df['Local Time'] = pd.to_datetime(ohlcv_df['Local Time'])
		ohlcv_df = ohlcv_df.set_index('Local Time').resample(self.resampling_unit).apply({'open': 'first',
																						  'high': 'max',
																						  'low': 'min',
																						  'close': 'last',
																						  'volume': 'sum'
																						  })
																	  
		ohlcv_df = ohlcv_df.between_time('9:00', '17:35').reset_index()
		
		return ohlcv_df
		
	def load_FOB_data(self, data_type: str):
		"""
		Loading FOB data among LOB, FO, CO and TIF
		"""
		data_file = [f for f in os.listdir(os.path.join(self.processed_path, data_type)) if self.to_process['ISIN'] == f.split('.')[0]]
		data = pd.read_parquet(os.path.join(self.processed_path, data_type, data_file))
		data = data.between_time('9:00', '17:35')
		
		return data
		
	def construct_LOB(self):
		"""
		Constructing LOB data with levels
		"""
		data = self.load_FOB_data('LOB')
		
		
	def array_process(self):
		"""
		Running script with slurm array jobs
		"""
		self.check_features()
		
		self.features_df = load_ohlcv()


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
	
	param = vars(args)
	
	reg = regression(args.job_id)

	if args.slurm_array:
		reg.array_process()