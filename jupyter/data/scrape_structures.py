#!/bin/env python3

import os
import re
import pandas as pd
import argparse
from mp_api.client import MPRester


# Defaule Materials Project API Key
MP_API_KEY = os.environ['MP_API_KEY']

# Default paths to data directories:
SUPERCON_CSV = './supercon/supercon1_data.csv'
PTABLE_CSV = './periodictable/PeriodicTable.csv'
STRUCTURE_EXPORT_DIR = './structures/unitcell'

parser = argparse.ArgumentParser(
    description = 'Scrapes the Materials Project database for superconductor unit cell structures.'
)

parser.add_argument('--supercon_csv', default=SUPERCON_CSV, required=False)
parser.add_argument('--ptable_csv', default=PTABLE_CSV, required=False)
parser.add_argument('--structure_export_dir', default=STRUCTURE_EXPORT_DIR, required=False)
parser.add_argument('--family', default='Metallic', required=False)
parser.add_argument('--max_ehull', type=float, default=0.0)


def build_formula_regex(elements):
    """ builds a formula parsing regex """
    mass_re = '([0-9]*\.[0-9]+|[0-9]+)'
    elem_re = '(' + '|'.join(elements) + ')'
    return re.compile(elem_re + mass_re)

def parse_formula_tokens(formula, regex):
    """ parses a checmical formula consisting of <elem><mass> pairs"""
    tokens = []
    for match in regex.finditer(formula):
        if match.group(1):
            tokens.append((match.group(1), float(match.group(2))))
        else:
            # assume 1.0 if no mass term:
            tokens.append((match.group(1), 1.0))
            
    return tokens

def make_formula_vector(tokens, elements):
    """converts a formula to a periodic table vector of elements"""
    v = np.zeros(len(elements))
    for t in tokens:
        v[elements.index(t[0])] += t[1]
    return v

def classify_superconductors(dataset_tokens):
    """ roughly classifies superconducting materials"""
    
    metals_set = set([ 'Sc', 'Ti', 'V', 'Ni', 
               'Cu','Zn', 'Pb', 'Zr', 'Nb', 
               'Mo', 'Pd', 'Ag', 'In', 'Sn',
               'Ir', 'Pt', 'Au', 'Hg', 'Pb' ])

    cuprate_set = []
    iron_based_set = []
    elemental_set = []
    metallic_alloy_set = []
    other_set = []

    for item in dataset_tokens:
        elems = list(t[0] for t in item[0])
        
        # classify elementals:
        if len(elems) <= 1:
            elemental_set.append(item)
            
        # classify cuprates:
        elif ('Cu' in elems):
            cuprate_set.append(item)
            
        # classify iron-based:
        elif ('Fe' in elems):
            iron_based_set.append(item)
        
        # classify mettalic alloy
        elif (set(elems) | metals_set):
            metallic_alloy_set.append(item)
            
        # classify as other:
        else:
            other_set.append(item)

    # construct as a dict of superconductor families:
    families = {
        'Elemental' : elemental_set,
        'Metallic' : metallic_alloy_set,
        'Iron-Based' : iron_based_set,
        'Cuprate' : cuprate_set,
    }  
    return families

def get_unmixed_material_tokens(material):
    """ returns a list of base materials that comprise a material
        (i.e. if a material is doped, it will return a list of fully
         doped variants of the material) """
    fixed_elems = {}
    varied_elems = {}
    
    # iterate through (element, number) pairs to
    #  infer dopants and their structures:
    for elem, n in material[0]:
        if int(n) > 0:
            fixed_elems[elem] = int(n)
        if 0.0001 < (n - int(n)):
            varied_elems[elem] = (n - int(n))
    
    if len(varied_elems) <= 0:
        return [ material[0] ]
    
    
    # construct each "fully doped" variant of the material:
    unmixed_materials = []
    for ve, n in sorted(list(varied_elems.items()), key=lambda x: -x[1]):
        
        new_material = { elem : n for elem, n in fixed_elems.items() }
        if ve not in new_material:
            new_material[ve] = 0
        new_material[ve] += 1    
        unmixed_materials.append(list(new_material.items()))
        
    return unmixed_materials

def formula_from_tokens(tokens):
    tokens = sorted(tokens)
    formula = ''
    for elem, n in tokens:
        formula += str(elem)
        if (n != 1):
            formula += str(n)
    return formula    

def main():

    args = parser.parse_args()

      # load PTable and Supercon datasets:
      # load data into pandas dataframe:
    supercon_df = pd.read_csv(args.supercon_csv)

    # separate out known T_c from unknown T_c data:
    known_tc = (supercon_df.Tc != 0)
    supercon_df['KnownTc'] = known_tc
    ptable_df = pd.read_csv(args.ptable_csv)


    # Roughly classify superconductor materials:
    elements = [ e.strip() for e in ptable_df.Symbol if e ]
    form_re = build_formula_regex(elements)
    data_idx = supercon_df.KnownTc

    dataset_tokens = [ 
        (parse_formula_tokens(form, form_re), (form,tc) ) 
        for form, tc in zip(supercon_df.name[data_idx],
                        supercon_df.Tc[data_idx])
    ]

    families = classify_superconductors(dataset_tokens)
    # export_materials = families['Metallic']
    export_materials = families[args.family]    

    with MPRester(MP_API_KEY) as mpr:
        for material in export_materials:
            formula = material[1][0]
                        
            material_tokens = get_unmixed_material_tokens(material)
            for tokens in material_tokens:
                elems = [ t[0] for t in tokens ]
                unmixed_form = formula_from_tokens(tokens)         
                export_path = os.path.join(args.structure_export_dir,
                                          f'{formula}__{unmixed_form}.poscar')
                
                if os.path.exists(export_path):
                    continue
                docs = mpr.summary.search(
                                      formula=unmixed_form, 
                                      elements=elems, 
                                      energy_above_hull = (0.0, 0.0),
                                      fields=['material_id', 'structure'])
                
                

                if len(docs) > 0:
                    if len(docs) > 1:
                        print('Multiple Stable Structures: ', docs)
                        continue
                    else:
                        docs[0].structure.to(fmt='poscar', filename=export_path)
                        print('Exported to:', export_path)
            
        

if __name__ == '__main__':
    main()
