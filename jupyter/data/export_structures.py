#!/bin/env python3

import os
import re
import json
from glob import glob

import pandas as pd
from ase.io import read, write
from tqdm import tqdm

SUPERCON_CSV = './supercon/supercon1_data.csv'

PTABLE_CSV = './periodictable/PeriodicTable.csv'

TRAJ_DATA_DIRS = [
    './structures/known',
    #'./structures/relaxed'
]

ALIGNN_DATA_EXPORT_DIR = './structures/alignn_data'
PROP_FILENAME = 'id_prop.csv'

ALIGNN_CONFIG_FILENAME = 'config.json'
DEFAULT_ALIGNN_CONFIG = {
    "version": "112bbedebdaecf59fb18e11c929080fb2f358246",
    "dataset": "user_data",
    "target": "target",
    "atom_features": "cgcnn",
    "neighbor_strategy": "k-nearest",
    "id_tag": "jid",
    "random_seed": 123,
    "classification_threshold": None,
    "n_val": None,
    "n_test": None,
    "n_train": None,
    "train_ratio": 0.8,
    "val_ratio": 0.1,
    "test_ratio": 0.1,
    "target_multiplication_factor": None,
    "epochs": 20,
    "batch_size": 8,
    "weight_decay": 1e-05,
    "learning_rate": 0.001,
    "filename": "sample",
    "warmup_steps": 2000,
    "criterion": "mse",
    "optimizer": "adamw",
    "scheduler": "onecycle",
    "pin_memory": False,
    "save_dataloader": False,
    "write_checkpoint": True,
    "write_predictions": True,
    "store_outputs": True,
    "progress": True,
    "log_tensorboard": False,
    "standard_scalar_and_pca": False,
    "use_canonize": True,
    "num_workers": 0,
    "cutoff": 8.0,
    "max_neighbors": 12,
    "keep_data_order": False,
    "model": {
        "name": "alignn",
        "alignn_layers": 4,
        "gcn_layers": 4,
        "atom_input_features": 92,
        "edge_input_features": 80,
        "triplet_input_features": 40,
        "embedding_features": 64,
        "hidden_features": 256,
        "output_features": 1,
        "link": "identity",
        "zero_inflated": False,
        "classification": False
    }
}

def build_formula_regex(elements):
    """ builds a formula parsing regex """
    mass_re = '([0-9]*\.[0-9]+|[0-9]+)'
    elem_re = '(' + '|'.join(elements) + ')'
    return re.compile(elem_re + mass_re)

def parse_formula_tokens(formula, regex):
    """ parses a chemical formula consisting of <elem><n> pairs"""
    tokens = []
    for match in regex.finditer(formula):
        if match.group(1):
            tokens.append((match.group(1), float(match.group(2))))
        else:
            # assume 1.0 if no 'n' term:
            tokens.append((match.group(1), 1.0))
            
    return tokens

def compile_structures(traj_data, supercon_data):
    pass    
        

def main():
   
    
    # load PTable and Supercon datasets:
    ptable_df = pd.read_csv(PTABLE_CSV)
    
    # load data into pandas dataframe:
    supercon_df = pd.read_csv(SUPERCON_CSV)

    # separate out known T_c from unknown T_c data:
    known_tc = (supercon_df.Tc != 0)
    supercon_df['KnownTc'] = known_tc

    # pase supercon formulas: 
    elements = [ e.strip() for e in ptable_df.Symbol if e ]
    form_re = build_formula_regex(elements)
    data_idx = supercon_df.KnownTc

    supercon_dataset = { 
        tuple(sorted(parse_formula_tokens(form, form_re))): (form,tc)
        for form, tc in zip(supercon_df.name[data_idx],
                            supercon_df.Tc[data_idx])
    }


    # construct export dataset as follows:
    #  formula -> (x = exported poscar filename, y = Tc value)
    print(f'Exporting data to: {ALIGNN_DATA_EXPORT_DIR}')
    export_dataset = {}
    for traj_dir in TRAJ_DATA_DIRS:
        traj_files = glob(os.path.join(traj_dir,'*.traj'))
        
        for filepath in tqdm(traj_files):
            _, name = os.path.split(filepath)
            formula = os.path.splitext(name)[0].split('__')[0]
            tokens = tuple(sorted(parse_formula_tokens(formula, form_re)))

            # if a match is found, export the data into POSCAR format:
            if tokens in supercon_dataset:
                export_file = f'{formula}.poscar'
                export_filepath = os.path.join(ALIGNN_DATA_EXPORT_DIR, export_file)
                export_dataset[formula] = (export_file, supercon_dataset[tokens][1])
                atoms = read(filepath)
                atoms.write(export_filepath)
                
            else:
                print('Unrecognized file: ', filepath)

    # write property CSV file:
    prop_csv_path = os.path.join(ALIGNN_DATA_EXPORT_DIR, PROP_FILENAME)
    with open(prop_csv_path, 'w+') as f:
        for export_file, Tc in export_dataset.values():
            f.write(', '.join([export_file, f'{Tc}']) + '\n')
    print(f'Saved properties to {prop_csv_path}')

    # write alignn config file:
    config_path = os.path.join(ALIGNN_DATA_EXPORT_DIR, ALIGNN_CONFIG_FILENAME)
    with open(config_path, 'w+') as f:
        f.write(json.dumps(DEFAULT_ALIGNN_CONFIG, indent=4))
    print(f'Saved ALIGNN configuration to {config_path}')
    

if __name__ == '__main__':
    main()
