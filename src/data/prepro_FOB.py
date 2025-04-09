#packages
import os, sys
import zipfile
import pandas as pd
import numpy as np
import argparse
from fastparquet import write

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from FOBDBM import FOBDataBaseManagement as fobdm

class FOBPreprocessor:
	"""
	Preprocessing FOB to create LOB.

	Args:
		job_id (int, optionnal): Slurm job ID, Default=0.
	"""
	def __init__(self, job_id: int = 0, resampling_unit: str = '5min'):
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
		while os.path.basename(self.root_path) != 'PhD_article_1':
			self.root_path =  os.path.dirname(self.root_path)
		self.job_id = job_id
		self.raw_path = os.path.join(self.root_path,'data','raw','FOB')
		self.processed_path = os.path.join(self.root_path,'data','processed','FOB')
		self.processed_path_LOB = os.path.join(self.processed_path,'LOB')
		self.processed_path_FO = os.path.join(self.processed_path,'FO')
		self.processed_path_CO = os.path.join(self.processed_path,'CO')
		self.processed_path_TIF = os.path.join(self.processed_path,'TIF')
		self.fobdm = fobdm(self.job_id)
		self.FOB = pd.DataFrame()
		self.LOB = None
		self.FO = None
		self.CO = None
		self.TIF = None
		self.filename_tmp = None
		self.filename = None
		self.filename_zip = None
		self.file = ''
		self.isin = ''
		self.resampling_unit = resampling_unit
		
	def load_FOB(self):
		"""
		Load FOB file into a DataFrame.
		
		Attributes:
			raw_path (str): Path of the repository with raw data of FOB /data/raw/FOB/.
			file (str): Name of the FOB file in process.
			FOB (DataFrame): Store the updated FOB DataFrame.
			isin (str): Name of the ISIN in process.
			
		Args:
			None: This method does not require args.

		Returns:
			None: This method does not return anything.
		
		Raises:
			None: This method does not raise error.
		"""
		chunks = []
		for chunk in pd.read_csv(os.path.join(self.raw_path, os.path.splitext(self.file)[0]), 
								header=0, 
								low_memory=False, 
								chunksize=10000, 
								usecols=['isin',
										'event_date',
										'event_time_cet',
										'order_id',
										'order_event_type',
										'order_side',
										'order_price',
										'order_size',
										'order_type',
										'time_in_force', 
										'trade_size', 
										'trade_price']):
			chunk = chunk[(chunk['isin'] == self.isin)]

			chunk['event_time_cet'] = pd.to_datetime(chunk['event_date'] + ' ' + chunk['event_time_cet'])
			chunk = chunk.drop(columns=['event_date'])
			chunks.append(chunk)
			
		self.FOB = pd.concat(chunks)
		self.FOB['time_in_force'] = self.FOB['time_in_force'].astype(str)
	
	def shift_orders(self):
		"""
		Shift previous order price and size for the same order ID when 'Modify' or 'Cancel' event type.
		
		Attributes:
			FOB (DataFrame): Store the updated FOB DataFrame.
		
		Args:
			None: This method does not require args.

		Returns:
			None: This method does not return anything.
		
		Raises:
			None: This method does not raise error.
		"""
		mask = (self.FOB['order_event_type'] == 'New') | (self.FOB['order_event_type'] == 'Reload')
		self.FOB['initial_index'] = self.FOB.index
		self.FOB = pd.concat([self.FOB[mask], self.FOB[~mask]], ignore_index=True)
		
		self.FOB['previous_price'] = self.FOB.groupby('order_id')['order_price'].shift(1)
		self.FOB['previous_size'] = self.FOB.groupby('order_id')['order_size'].shift(1)
		
		self.FOB = self.FOB.sort_values('initial_index').set_index('initial_index')
		self.FOB.name = None

			
	def resample_FOB_LOB(self, data, price: str, size: str, to_add: bool = True):
		"""
		Resample the FOB for add/subtract sizes.
		
		Attributes:
			FOB (DataFrame): FOB DataFrame.
			resampling_unit (str): Rule of resampling for the FOB.
		
		Args:
			None: This method does not require args.

		Returns:
			resample_df (DataFrame): Resampled FOB dataframe with limit orders only.
		
		Raises:
			None: This method does not raise error.
		"""
		resample_df = data.copy()
		
		resample_df = resample_df[['event_time_cet', 'order_side'] + [price, size]]
		
		if to_add == False:
			resample_df[size] *= -1

		resample_df.set_index('event_time_cet', inplace=True)
		resample_df = resample_df.groupby(['order_side', price]).resample(self.resampling_unit).sum()[size].to_frame()

		resample_df = resample_df[~resample_df.isna().any(axis=1)]
		resample_df = resample_df.reset_index().groupby(['event_time_cet', 'order_side', price], as_index=False).last()
		
		resample_df.columns = ['event_time_cet', 'side', 'price', 'size']
		
		return resample_df
	
	def construct_LOB(self):
		"""
		Contruct LOB dataframe.
		
		Attributes:
			FOB (DataFrame): FOB DataFrame.
			LOB (DataFrame): LOB DataFrame.
			filename_tmp (str): Name of the temporary csv file with LOB dataframe.
			processed_path (str): Path of the repository with processed data of FOB /data/processed/FOB/.
		
		Args:
			None: This method does not require args.

		Returns:
			None: This method does not return anything.
		
		Raises:
			None: This method does not raise error.
		"""
		data = self.FOB.copy()
		data = data.loc[(data['order_type'] == 'Limit') & (data['time_in_force'] == '0')]
		LOB_add = self.resample_FOB_LOB(data=data, price='order_price', size='order_size')
		LOB_sub = self.resample_FOB_LOB(data=data, price='previous_price', size='previous_size', to_add=False)
		resamp_FOB_LOB = pd.concat([LOB_add, LOB_sub], ignore_index=True).groupby(['event_time_cet', 'side', 'price'], as_index=False).sum()
		
		time = resamp_FOB_LOB['event_time_cet'].sort_values().unique().tolist()
		lentime = len(time)

		self.LOB = pd.DataFrame(columns=['price', 'size', 'side'])
		
		if self.filename_tmp in os.listdir(self.processed_path):
			self.LOB = pd.read_parquet(os.path.join(self.processed_path, self.filename_tmp))
			self.LOB.index = pd.to_datetime(self.LOB.index)
			ls_t = pd.to_datetime(self.LOB.index.unique().tolist())
			
			for t in reversed(ls_t):
				if t in time:
					last_t = t
					break
					
			time = [x for x in time if x > last_t]
			
			if not time:
				print(f'{self.isin} already processed in {self.file}')
				return 0
				
			self.LOB = self.LOB.loc[last_t]
			
		else:
			ls_t = False
	
		for t, block in resamp_FOB_LOB.groupby('event_time_cet'):
			
			if ls_t != False:
				if t in ls_t:
					print('t in ls_t')
					continue
			else:
				pass
			
			if time.index(pd.to_datetime(t)) % 100 == 0:
				print(f'{time.index(pd.to_datetime(t))} / {lentime}')
			
			tmp = block[['price', 'size', 'side']]

			self.LOB = self.LOB.reset_index(drop=True)

			self.LOB = pd.concat([self.LOB, tmp], ignore_index=True).groupby(['price', 'side'], as_index=False).sum()
			self.LOB = self.LOB[self.LOB['size'] != 0]
			self.LOB.index = pd.Index([t] * len(self.LOB))
			
			self.save_LOB(state='tmp')
		
		self.save_LOB(state='def')
		
	def construct_FO(self):
		"""
		Contruct FO dataframe.
		
		Attributes:
			FOB (DataFrame): FOB DataFrame.
			filename_zip (str): Name of the gzip file with the final parquet file with LOB/FO dataframe.
			processed_path (str): Path of the repository with processed data of FOB /data/processed/FOB/.
		
		Args:
			None: This method does not require args.

		Returns:
			None: This method does not return anything.
		
		Raises:
			None: This method does not raise error.
		"""
		self.FO = self.FOB.copy()
		self.FO = self.FO.loc[(self.FO['order_event_type'] == 'Fill'), ['event_time_cet','order_side','trade_size','trade_price']]

		self.FO.set_index('event_time_cet', inplace=True)
		self.FO = self.FO.groupby(['order_side', 'trade_price']).resample(self.resampling_unit).sum()['trade_size'].to_frame()

		self.FO = self.FO[~self.FO.isna().any(axis=1)]
		self.FO = self.FO.reset_index().groupby(['event_time_cet', 'order_side', 'trade_price'], as_index=False).last()
		self.FO = self.FO[self.FO['trade_size'] != 0].set_index('event_time_cet')
		
		write(os.path.join(self.processed_path_FO, self.filename_zip), self.FO, compression='GZIP', append=False)
		
	def save_LOB(self, state: str = 'tmp'):
		"""
		Save LOB as tmp or final csv file.
		
		Attributes:
			LOB (DataFrame): LOB DataFrame.
			filename_tmp (str): Name of the temporary csv file with LOB dataframe.
			filename_zip (str): Name of the gzip file with the final parquet file with LOB/FO dataframe.
			processed_path_LOB (str): Path of the repository with processed data of LOB /data/processed/FOB/ where the LOB is saved.
			
		Args:
			state (str, optionnal): tmp or final version of the LOB csv file, Default='tmp'.

		Returns:
			None: This method does not return anything.
		
		Raises:
			None: This method does not raise error.
		"""
		if state == 'tmp':
			if self.filename_tmp not in os.listdir(self.processed_path_LOB):
				write(os.path.join(self.processed_path_LOB, self.filename_tmp), self.LOB, compression='GZIP', append=False)
			else:
				write(os.path.join(self.processed_path_LOB, self.filename_tmp), self.LOB, compression='GZIP', append=True)
				
		if state == 'def':			
			os.rename(os.path.join(self.processed_path_LOB, self.filename_tmp), os.path.join(self.processed_path_LOB, self.filename_zip))
			
	def construct_CO(self):
		"""
		Contruct Cancel orders dataframe.
		
		Attributes:
			FOB (DataFrame): FOB DataFrame.
			filename_zip (str): Name of the gzip file with the final parquet file with LOB/FO dataframe.
			processed_path (str): Path of the repository with processed data of FOB /data/processed/FOB/.
		
		Args:
			None: This method does not require args.

		Returns:
			None: This method does not return anything.
		
		Raises:
			None: This method does not raise error.
		"""
		self.CO = self.FOB.copy()
		ls_order = self.CO.loc[self.CO['order_event_type'] == 'Cancel', 'order_id']
		self.CO = self.CO.loc[self.CO['order_id'].isin(ls_order), ['order_id', 'order_event_type', 'event_time_cet', 'order_side', 'order_size']]

		def attrib_last_size(chunk):
			chunk.loc[chunk['order_event_type'] == 'Cancel', 'order_size'] = chunk.sort_values('event_time_cet').loc[chunk['order_event_type'] != 'Cancel', 'order_size'].iloc[-1]
			return chunk.loc[chunk['order_event_type'] == 'Cancel']

		self.CO = self.CO.groupby(['order_id'], as_index=False).apply(attrib_last_size)
		self.CO.set_index('event_time_cet', inplace=True)
		self.CO = self.CO.groupby(['order_side']).resample('5min').sum()['order_size'].to_frame()
		self.CO = self.CO.reset_index().groupby(['event_time_cet', 'order_side'], as_index=False).last()
		self.CO = self.CO[self.CO['order_size'] != 0].set_index('event_time_cet')
		
		write(os.path.join(self.processed_path_CO, self.filename_zip), self.CO, compression='GZIP', append=False)
		
	def resample_TIF_LOB(self, data, price: str, size: str, to_add: bool = True):
		"""
		Resample the FOB for add/subtract sizes.
		
		Attributes:
			FOB (DataFrame): FOB DataFrame.
			resampling_unit (str): Rule of resampling for the FOB.
		
		Args:
			None: This method does not require args.

		Returns:
			resample_df (DataFrame): Resampled FOB dataframe with limit orders only.
		
		Raises:
			None: This method does not raise error.
		"""
		resample_df = data.copy()
		
		resample_df = resample_df[['event_time_cet', 'order_side', 'order_type', 'time_in_force'] + [price, size]]
		
		if to_add == False:
			resample_df[size] *= -1

		resample_df.set_index('event_time_cet', inplace=True)
		resample_df = resample_df.groupby(['order_side', 'order_type', 'time_in_force', price]).resample(self.resampling_unit).sum()[size].to_frame()

		resample_df = resample_df[~resample_df.isna().any(axis=1)]
		resample_df = resample_df.reset_index().groupby(['event_time_cet', 'order_side', 'order_type', 'time_in_force', price], as_index=False).last()
		
		resample_df.columns = ['event_time_cet', 'side', 'order_type', 'time_in_force', 'price', 'size']
		
		return resample_df
		
	def construct_TIF(self):
		"""
		Contruct TIF dataframe.
		
		Attributes:
			FOB (DataFrame): FOB DataFrame.
			LOB (DataFrame): LOB DataFrame.
			filename_tmp (str): Name of the temporary csv file with LOB dataframe.
			processed_path (str): Path of the repository with processed data of FOB /data/processed/FOB/.
		
		Args:
			None: This method does not require args.

		Returns:
			None: This method does not return anything.
		
		Raises:
			None: This method does not raise error.
		"""
		data = self.FOB.copy()
		data = data.loc[data['time_in_force'] != '0']
		LOB_add = self.resample_TIF_LOB(data=data, price='order_price', size='order_size')
		LOB_sub = self.resample_TIF_LOB(data=data, price='previous_price', size='previous_size', to_add=False)
		resamp_FOB_LOB = pd.concat([LOB_add, LOB_sub], ignore_index=True).groupby(['event_time_cet', 'side', 'order_type', 'time_in_force', 'price'], as_index=False).sum()
		
		TIF_init = pd.DataFrame({'side':['Buy','Buy','Buy','Buy','Sell','Sell','Sell','Sell'],
							'order_type':['Market','Market','Limit','Limit','Market','Market','Limit','Limit'],
							'time_in_force':['Valid for Uncrossing',
											 'Valid for Closing',
											 'Valid for Uncrossing',
											 'Valid for Closing',
											 'Valid for Uncrossing',
											 'Valid for Closing',
											 'Valid for Uncrossing',
											 'Valid for Closing'],
							'size':[0,0,0,0,0,0,0,0]})
	
		for t, block in resamp_FOB_LOB.groupby('event_time_cet'):			
			tmp = block[['side', 'order_type', 'time_in_force', 'size']]

			TIF_init = pd.concat([TIF_init, tmp], ignore_index=True).groupby(['side', 'order_type', 'time_in_force'], as_index=False).sum()
			TIF_init.index = pd.Index([t] * len(TIF_init))
			self.TIF = pd.concat([self.TIF, TIF_init])
		
		write(os.path.join(self.processed_path_TIF, self.filename_zip), self.TIF, compression='GZIP', append=False)
	
	def array_process(self, LOB_process: bool = True, Fill_order_process: bool = True, Cancel_order_process: bool = True, TIF_order_process: bool = True, **kwargs):
		"""
		Lauch FOB preprocessing from slurm array jobs.
		
		Attributes:
			file (str): Name of the FOB file in process.
			isin (str): Name of the ISIN in process.
			filename_tmp (str): Name of the temporary csv file with LOB dataframe.
			filename_zip (str): Name of the zip file with the final csv file with LOB dataframe.
			processed_path (str): Path of the repository with processed data of FOB /data/processed/FOB/.
		
		Args:
			None: This method does not require args.

		Returns:
			None: This method does not return anything.
		
		Raises:
			None: This method does not raise error.
		"""
		dict_k = kwargs
		self.file, self.isin = self.fobdm.main(**dict_k)
		print(self.file, self.isin, LOB_process, Fill_order_process, Cancel_order_process, TIF_order_process, dict_k)
		
		if LOB_process:
			date = os.path.splitext(os.path.splitext(self.file)[0])[0].split('_')[-1]
			self.filename_tmp = f'{self.isin}_{date}_LOB_tmp.parquet.gzip'
			self.filename_zip = f'{self.isin}_{date}_LOB.parquet.gzip'
			
			if self.filename_zip in os.listdir(self.processed_path_LOB):
				print('Already done')
				pass
				
			else:
				print('Start LOB process')
				if self.FOB.empty:
					self.load_FOB()
					
				self.shift_orders()
				self.construct_LOB()
			
			self.fobdm.terminate(data_type='LOB')
			
		if Fill_order_process:
			date = os.path.splitext(os.path.splitext(self.file)[0])[0].split('_')[-1]
			self.filename_zip = f'{self.isin}_{date}_FO.parquet.gzip'
			
			if self.filename_zip in os.listdir(self.processed_path_FO):
				print('Already done')
				pass
				
			else:
				print('Start FO process')
				if self.FOB.empty:
					self.load_FOB()

				self.construct_FO()
			
			self.fobdm.terminate(data_type='FO')
			
		if Cancel_order_process:
			date = os.path.splitext(os.path.splitext(self.file)[0])[0].split('_')[-1]
			self.filename_zip = f'{self.isin}_{date}_CO.parquet.gzip'
			
			if self.filename_zip in os.listdir(self.processed_path_CO):
				print('Already done')
				pass
				
			else:
				print('Start CO process')
				if self.FOB.empty:
					self.load_FOB()

				self.construct_CO()
			
			self.fobdm.terminate(data_type='CO')
			
		if TIF_order_process:
			date = os.path.splitext(os.path.splitext(self.file)[0])[0].split('_')[-1]
			self.filename_zip = f'{self.isin}_{date}_TIF.parquet.gzip'
			
			if self.filename_zip in os.listdir(self.processed_path_TIF):
				print('Already done')
				pass
				
			else:
				print('Start TIF process')
				if self.FOB.empty:
					self.load_FOB()
					
				self.shift_orders()
				self.construct_TIF()
			
			self.fobdm.terminate(data_type='TIF')
			
	def concat_data(self, LOB_process: bool = True, Fill_order_process: bool = True, Cancel_order_process: bool = True, TIF_order_process: bool = True):
		"""
		Concatenate each LOB/FO files by isin.
		
		Attributes:
			file (str): Name of the FOB file in process.
			isin (str): Name of the ISIN in process.
			filename_tmp (str): Name of the temporary csv file with LOB dataframe.
			filename_zip (str): Name of the zip file with the final csv file with LOB dataframe.
			processed_path (str): Path of the repository with processed data of FOB /data/processed/FOB/.
		
		Args:
			None: This method does not require args.

		Returns:
			None: This method does not return anything.
		
		Raises:
			None: This method does not raise error.
		"""
		if LOB_process:
			files = [i for i in os.listdir(self.processed_path_LOB) if 'final' not in i]
			isin_ls = list(set([i.split('_')[0] for i in files]))
			
			for isin in isin_ls:
				df = pd.DataFrame()
				for f in [file for file in files if isin in file]:
					data = pd.read_parquet(os.path.join(self.processed_path_LOB, f))
					df = pd.concat([df, data])
					os.remove(os.path.join(self.processed_path_LOB, f))
					
				write(os.path.join(self.processed_path_LOB, f'{isin}_final_LOB.parquet.gzip'), df, compression='GZIP', append=False)
			
		if Fill_order_process:
			files = [i for i in os.listdir(self.processed_path_FO) if 'final' not in i]
			isin_ls = list(set([i.split('_')[0] for i in files]))
			
			for isin in isin_ls:
				df = pd.DataFrame()
				for f in [file for file in files if isin in file]:
					data = pd.read_parquet(os.path.join(self.processed_path_FO, f))
					df = pd.concat([df, data])
					os.remove(os.path.join(self.processed_path_FO, f))
					
				write(os.path.join(self.processed_path_FO, f'{isin}_final_FO.parquet.gzip'), df, compression='GZIP', append=False)
				
		if Cancel_order_process:
			files = [i for i in os.listdir(self.processed_path_CO) if 'final' not in i]
			isin_ls = list(set([i.split('_')[0] for i in files]))
			
			for isin in isin_ls:
				df = pd.DataFrame()
				for f in [file for file in files if isin in file]:
					data = pd.read_parquet(os.path.join(self.processed_path_CO, f))
					df = pd.concat([df, data])
					os.remove(os.path.join(self.processed_path_CO, f))
					
				write(os.path.join(self.processed_path_CO, f'{isin}_final_CO.parquet.gzip'), df, compression='GZIP', append=False)
				
		if TIF_order_process:
			files = [i for i in os.listdir(self.processed_path_CO) if 'final' not in i]
			isin_ls = list(set([i.split('_')[0] for i in files]))
			
			for isin in isin_ls:
				df = pd.DataFrame()
				for f in [file for file in files if isin in file]:
					data = pd.read_parquet(os.path.join(self.processed_path_TIF, f))
					df = pd.concat([df, data])
					os.remove(os.path.join(self.processed_path_TIF, f))
					
				write(os.path.join(self.processed_path_CO, f'{isin}_final_TIF.parquet.gzip'), df, compression='GZIP', append=False)
		
	
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
	parser.add_argument('--LOB_process', '-l', type=str2bool, default=True)
	parser.add_argument('--Fill_order_process', '-fo', type=str2bool, default=True)
	parser.add_argument('--Cancel_order_process', '-co', type=str2bool, default=True)
	parser.add_argument('--TIF_order_process', '-tif', type=str2bool, default=True)
	parser.add_argument('--concat', '-c', type=str2bool, default=False)
	args = parser.parse_args()
	
	fobp = FOBPreprocessor(args.job_id)    

	if args.slurm_array:
		fobp.array_process(**vars(args))
		
	if args.concat:
		fobp.concat_data(**vars(args))