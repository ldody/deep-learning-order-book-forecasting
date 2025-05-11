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

path = os.path.dirname(os.path.abspath(__file__))
root_path = path
while '.venv' not in os.listdir(root_path):
	root_path =  os.path.dirname(root_path)
path_model = os.path.join(root_path,'model')

ls_log = [f for f in os.listdir(path_model) if os.path.splitext(f)[1] == '.log']

for f in ls_log:
	isin, data, model = f.split('.')[0].split('_')
	
	
	optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
	STUDY_NAME = f'{isin}_{data}_{model}_optuna_study'
	DB_PATH = os.path.join(path_model, f'{isin}_{data}_{model}_optimization.log')
	storage = optuna.storages.JournalStorage(
		optuna.storages.journal.JournalFileBackend(DB_PATH),
	)
	
	try:
		study = optuna.load_study(storage=storage, study_name=STUDY_NAME)
		df = study.trials_dataframe()
		
		if list(set(df['state'].tolist())) == ['RUNNING']:
			optuna.delete_study(storage=storage, study_name=STUDY_NAME)
		
		else:
			pass
	
	except:
		traceback.print_exc()
		
