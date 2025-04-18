#packages
import os, sys
import zipfile
import pandas as pd
import numpy as np
from fastparquet import write
import argparse

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
		while '.venv' not in os.listdir(self.root_path):
			self.root_path =  os.path.dirname(self.root_path)
		self.job_id = job_id
		self.resampling_unit = resampling_unit
		self.data_path = os.path.join(self.root_path,'data')
		self.ohlcv_path = os.path.join(self.data_path,'raw','OHLCV')
		self.processed_path = os.path.join(self.data_path,'processed')
		self.features_path = os.path.join(self.processed_path,'features')
		self.FOB_path = os.path.join(self.processed_path,'FOB')
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
		ohlcv_file = [f for f in os.listdir(self.ohlcv_path) if self.to_process['RIC'] in f][0]
		ohlcv_df = pd.read_csv(os.path.join(self.ohlcv_path, ohlcv_file), index_col='Local Time')
		ohlcv_df.index = pd.to_datetime(ohlcv_df.index)
		ohlcv_df.ffill(inplace=True)
		ohlcv_df = ohlcv_df.resample(self.resampling_unit).apply({'Open': 'first',
																  'High': 'max',
																  'Low': 'min',
																  'Close': 'last',
																  'Volume': 'sum'
																  })
																	  
		ohlcv_df = ohlcv_df.between_time('9:00', '17:35')
		ohlcv_df = ohlcv_df.dropna()
		
		return ohlcv_df
		
	def load_FOB_data(self, data_type: str):
		"""
		Loading FOB data among LOB, FO, CO and TIF
		"""
		data_file = [f for f in os.listdir(os.path.join(self.FOB_path, data_type)) if self.to_process['ISIN'] in f][0]
		data = pd.read_parquet(os.path.join(self.FOB_path, data_type, data_file))
		data = data.between_time('9:00', '17:35')
		
		return data
		
	def construct_LOB(self, nb_levels: int = 60):
		"""
		Constructing LOB data with levels
		"""
		data = self.load_FOB_data('LOB')
		
		final_b = pd.DataFrame()
		final_s = pd.DataFrame()
		final = pd.DataFrame()
		
		ls_b = []
		ls_s = []
		
		tick_p = self.to_process['Tick_step']
		var_p = self.to_process['5min']
		coef = int(var_p/tick_p/nb_levels*2)
		w_int = tick_p * coef

		for _, chunk in data.loc[data['side'] == 'Buy'].groupby('index'):

			bins = [chunk['price'].max()-v*w_int for v in range(nb_levels+1)]

			chunk['interval'] = pd.cut(chunk['price'], bins=bins[::-1], right=True)

			chunk['price'] = chunk['interval'].apply(lambda x: x.left)

			chunk = chunk.reset_index()

			chunk = chunk.groupby(['index','side','price'])['size'].sum().reset_index()
			chunk = chunk.sort_values('price', ascending=False).reset_index()
			chunk['rank'] = chunk.groupby('index')['price'].rank(ascending=False)
			chunk['idx'] = 0
			
			test = chunk.pivot(index=['index'], columns=['rank','side'], values=['price', 'size']).T.reset_index()
			test = test.groupby(['rank','side','level_0']).last()
			final_b = pd.concat([final_b, test.T])
			ls_b.append(chunk['size'].sum())
			
		for _, chunk in data.loc[data['side'] == 'Sell'].groupby('index'):

			bins = [chunk['price'].min()+v*w_int for v in range(nb_levels+1)]

			chunk['interval'] = pd.cut(chunk['price'], bins=bins, right=False)

			chunk['price'] = chunk['interval'].apply(lambda x: x.right)

			chunk = chunk.reset_index()

			chunk = chunk.groupby(['index','side','price'])['size'].sum().reset_index()
			chunk = chunk.sort_values('price', ascending=True).reset_index()
			chunk['rank'] = chunk.groupby('index')['price'].rank(ascending=True)
			chunk['idx'] = 0
			
			test = chunk.pivot(index=['index'], columns=['rank','side'], values=['price', 'size']).T.reset_index()
			test = test.groupby(['rank','side','level_0']).last()
			final_s = pd.concat([final_s, test.T])
			ls_s.append(chunk['size'].sum())
			
		final = pd.concat([final_b.T, final_s.T], axis=0).groupby(['rank','side','level_0']).last().T
		final.columns = ['{}_{}_{}'.format(side.lower(), key, int(rank)) for rank, side, key in final.columns]
		final['buy_liquidity'] = ls_b
		final['sell_liquidity'] = ls_s
		final['total_liquidity'] = final[['buy_liquidity','sell_liquidity']].sum(axis=1)

		return final
		
	def construct_FO(self, nb_levels: int = 5):
		"""
		Constructing FO data with levels
		"""
		df = self.load_FOB_data('FO')
		
		df = df[df['order_side'] == 'Buy'].drop(['order_side'], axis=1)
		a = df.between_time('9:00', '17:35').reset_index().groupby(['event_time_cet'], as_index=False).sum()
		df = df.between_time('9:00', '17:35').reset_index().groupby(['event_time_cet'], as_index=False).apply(lambda x: x.nlargest(nb_levels, 'trade_size'))
		df = df.set_index(['event_time_cet'])

		tmp = df.index.value_counts().to_frame()
		tmp['count'] = nb_levels - tmp['count']
		tmp = tmp[tmp['count'] != 0]
		ls = []
		for idx, c in tmp.iterrows():
			ls.insert(c['count'], idx)

		tmp = pd.DataFrame(index=ls)
		tmp.index.name = df.index.name
		df = pd.concat([df,tmp]).reset_index().fillna(0)
		df['idx'] = df.groupby('event_time_cet')['trade_size'].rank(ascending=False, method='first', na_option='top')
		df = df.pivot(index=['event_time_cet'], columns=['idx'], values=['trade_price', 'trade_size']).T.reset_index()
		df = df.groupby(['idx','level_0']).last().T.fillna(0)
		df.columns = ['FO_{}_{}'.format(data_t.split('_')[1], int(rank)) for rank, data_t in df.columns]
		
		return df
		
	def construct_CO(self):
		"""
		Constructing CO data
		"""
		df = self.load_FOB_data('CO')
		
		df = df.between_time('9:00', '17:35').reset_index().groupby(['event_time_cet','order_side']).last()
		df = df.unstack()
		df.columns = ['{}_CO_size'.format(side.lower()) for _, side in df.columns]
		
		return df
		
	def construct_TIF(self):
		"""
		Constructing TIF data
		"""
		df = self.load_FOB_data('TIF')
		
		df = df.between_time('9:00', '17:35').reset_index().groupby(['index','side','order_type','time_in_force']).last()
		df = df.unstack(['side','order_type','time_in_force'])
		df.columns = ['{}_{}_{}_{}'.format(side.lower(), o_type, tif, size) for size, side, o_type, tif in df.columns]
		
		return df
		
	def array_process(self):
		"""
		Running script with slurm array jobs
		"""
		self.check_features()
		
		self.features_df = self.load_ohlcv()
		print(self.features_df)
		self.features_df = pd.concat([self.features_df, self.construct_LOB()], ignore_index=False, axis=1)
		#print(self.features_df)
		self.features_df = pd.concat([self.features_df, self.construct_FO()], ignore_index=False, axis=1).dropna(subset=['Close']).fillna(0)
		self.features_df = pd.concat([self.features_df, self.construct_CO()], ignore_index=False, axis=1).dropna(subset=['Close']).fillna(0)
		self.features_df = pd.concat([self.features_df, self.construct_TIF()], ignore_index=False, axis=1).dropna(subset=['Close']).fillna(0)
		print(self.features_df)
		filename_zip = f'{self.to_process["ISIN"]}_features.parquet.gzip'
		write(os.path.join(self.features_path, filename_zip), self.features_df, compression='GZIP', append=False)
		
	def main(self):
		"""
		Running script
		"""
		for i, row in self.df_assets.iterrows():
			self.to_process = row
			self.features_df = self.load_ohlcv()
			self.features_df = pd.concat([self.features_df, self.construct_LOB()], ignore_index=False, axis=1)
			self.features_df = pd.concat([self.features_df, self.construct_FO()], ignore_index=False, axis=1).dropna(subset=['Close']).fillna(0)
			self.features_df = pd.concat([self.features_df, self.construct_CO()], ignore_index=False, axis=1).dropna(subset=['Close']).fillna(0)
			self.features_df = pd.concat([self.features_df, self.construct_TIF()], ignore_index=False, axis=1).dropna(subset=['Close']).fillna(0)
			filename_zip = f'{self.to_process["ISIN"]}_features.parquet.gzip'
			write(os.path.join(self.features_path, filename_zip), self.features_df, compression='GZIP', append=False)
			print(f'{i}/{len(self.df_assets)} done')


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
	
	bf = BuildFeatures(args.job_id)

	if args.slurm_array:
		bf.array_process()
		
	else:
		bf.main()