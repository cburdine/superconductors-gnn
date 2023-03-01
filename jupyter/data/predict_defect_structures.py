#!/bin/env python3

import os
import re
from fractions import Fraction
from multiprocessing import Pool
import random
import glob
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm
from ase.io import read
from ase.io.vasp import write_vasp
from ase.build import make_supercell, find_optimal_cell_shape
from ase.neighborlist import NeighborList
from ase.constraints import FixAtoms


PTABLE_CSV = './periodictable/PeriodicTable.csv'

UNITCELL_DATA_DIR = './structures/unitcell'
KNOWN_STRUCTURE_EXPORT_DIR = './structures/known'
DEFECT_STRUCTURE_EXPORT_DIR = './structures/defect'

parser = argparse.ArgumentParser(
    description='Predicts the structure of materials based on data acquired from the Materials Project database.'
)

parser.add_argument('--ptable_csv', default=PTABLE_CSV)
parser.add_argument('--unitcell_data_dir', default=UNITCELL_DATA_DIR)
parser.add_argument('--known_structure_export_dir', default=KNOWN_STRUCTURE_EXPORT_DIR)
parser.add_argument('--defect_structure_export_dir', default=DEFECT_STRUCTURE_EXPORT_DIR)
parser.add_argument('--processes', default=1)

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

def formula_from_tokens(tokens):
    """ Constructs a canonical formula from element tokens"""
    tokens = sorted(tokens)
    formula = ''
    for elem, n in tokens:
        formula += str(elem)
        if (n != 1):
            formula += str(n)
    return formula        

def get_unmixed_material_tokens(material_tokens):
    """ returns a list of base materials that comprise a material
        (i.e. if a material is doped, it will return a list of fully
         doped variants of the material) """
    fixed_elems = {}
    varied_elems = {}

    # iterate through (element, number) pairs to
    #  infer dopants and their structures:
    for elem, n in material_tokens:
        if int(n) > 0:
            fixed_elems[elem] = int(n)
        if 0.0001 < (n - int(n)):
            varied_elems[elem] = (n - int(n))

    if len(varied_elems) <= 0:
        return [ material_tokens ]
    
    # construct each "fully doped" variant of the material:
    unmixed_materials = []
    for ve, n in sorted(list(varied_elems.items()), key=lambda x: -x[1]):

        new_material = { elem : n for elem, n in fixed_elems.items() }
        if ve not in new_material:
            new_material[ve] = 0
        new_material[ve] += 1
        unmixed_materials.append(list(new_material.items()))

    return unmixed_materials # return materials (sorted by closest)

def substitute_elements(supercell, substitutions, dopants, random_choice=False):
    new_supercell = supercell.copy()
    
    if len(dopants) <= 0:
        return new_supercell, []
    
    assert(sum(x for _,x in substitutions) >= sum(x for _, x in dopants))
    subs_elements = {elem for elem, _ in substitutions}
    
    for dopant, n in dopants:
        sub_idxs = np.array([
            i for i, atom in enumerate(new_supercell) 
            if atom.symbol in subs_elements 
        ])
        
        if random_choice:
            chosen_sub_idxs = random.sample(list(sub_idxs), n)
        else:
            chosen_sub_idxs = sub_idxs[::len(sub_idxs)//n][:n]
        
        for idx in chosen_sub_idxs:
            new_supercell[idx].symbol = dopant
    
    
    return new_supercell, chosen_sub_idxs 

def find_least_integer_ratio(base_ratio_deltas, max_err=1e-5):

    for d in base_ratio_deltas.values():
        assert(abs(d) < 1.0)

    # partition and sort positive (added), negative (substituted) and zero (unchanged) elements:
    positive_elems = sorted([ (x,elem) for elem,x in base_ratio_deltas.items() if x > max_err ])   
    negative_elems = sorted([ (-x,elem) for elem,x in base_ratio_deltas.items() if x < -max_err ])
    zero_elems =     sorted([ (x,elem) for elem,x in base_ratio_deltas.items() if abs(x) <= max_err ])        
    approx_base_ratio_deltas = { elem : Fraction(0) for _,elem in zero_elems}
    
    # if formula is already an integer ratio, return:
    if len(positive_elems) == 0 or len(negative_elems) == 0:
        return { k : Fraction(0) for k in base_ratio_deltas }

    # generate a least integer ratio with the least concentrated added element having
    # a numerator of 1 in the ratio, and the rest of the ratios scaled proportionately.
    # This tweaks the actual proportions a bit, but results in a smaller supercell.
    denominator = int(1./positive_elems[0][0])
    numerator_sum = 0
    for x, elem in positive_elems:
        approx_x = int(round(x * denominator))
        numerator_sum += approx_x
        approx_base_ratio_deltas[elem] = Fraction(approx_x,denominator)

    for x, elem in negative_elems[:-1]:
        approx_x = -int(round(x * approx_denominator))
        numerator_sum += approx_x
        approx_base_ratio_deltas[elem] = Fration(approx_x, denominator)

    _, elem = negative_elems[-1]
    approx_base_ratio_deltas[elem] = Fraction(-numerator_sum,denominator)

    assert(sum(approx_base_ratio_deltas.values()) == 0)
    return approx_base_ratio_deltas

    

def generate_defect_supercells(tokens, base_tokens, atoms,
                               max_n_samples=1,
                               random=False,
                               formula=None,
                               min_concentration_error=0.01,
                               least_integer_ratio=True,
                               relaxation_R=2.0):
    
    base_ratios = { elem : n for elem, n in base_tokens }
    base_ratio_deltas = {}
    generated_supercells = []
    
    for elem, x in tokens:
        if elem not in base_ratios:
            base_ratio_deltas[elem] = x 
        else:
            base_ratio_deltas[elem] = (x - base_ratios[elem])

    # ignore if formula concentrations do not add up to 1:
    delta_sum = sum(base_ratio_deltas.values())
    if abs(delta_sum) > min_concentration_error:
        return []

    # If requested, reduce fractions to an approximate integer ratio (to avoid large supercells):
    if least_integer_ratio:
        base_ratio_deltas = find_least_integer_ratio(base_ratio_deltas)

    # here, we estimate supercell atoms = cell atoms * lcm(formula denominators)
    formula_denominators = [int(x.denominator) for x in base_ratio_deltas.values() if x != 0 ]
    if len(formula_denominators) > 0:
         
        # determine smallest supercell factor N needed according to:
        #  --->  N = numerator(lcm(x_denominators) , gcd(substitution_n_values)
        substitution_numerators = [ 
            int(n) for elem, n in base_ratios.items()
            if elem in base_ratio_deltas and base_ratio_deltas[elem] < 0
        ]
        n_supercells = np.lcm.reduce(formula_denominators)
        n_supercells //= np.gcd(n_supercells, np.gcd.reduce(substitution_numerators))
                
        P = find_optimal_cell_shape(atoms.cell, n_supercells, 'fcc')
        supercell = make_supercell(atoms, P)
        
        # determine dopants and substitutions that will be made to supercell:
        substitutions = []
        dopants = []
        for elem, x in base_ratio_deltas.items():
            if x < 0:
                substitutions.append((elem,int(np.ceil(-float(x)*n_supercells))))
            if x > 0:
                dopants.append((elem, int(np.ceil(float(x)*n_supercells))))
                
        # if substitutions can be made within the supercell, add them, else ignore them:
        print('Deltas:', base_ratio_deltas)
        if sum(x for _,x in substitutions) >= sum(x for _,x in dopants):
            for _ in range(max_n_samples):
                
                # generate supercell through random substitution:
                new_supercell, site_idxs = substitute_elements(supercell,
                                                          substitutions,
                                                          dopants,
                                                          random_choice=random)
                                           
                
                # set the relaxation radius:
                set_relaxation_radius(new_supercell, site_idxs, relaxation_R) 
                generated_supercells.append(new_supercell)
            
            return generated_supercells
        else:
            print('Failed to construct supercell:')
            print(f'Substitutions: {substitutions},\n Dopants: {dopants}')
            return []
    else:
        # if no dopants need to be added, return just the atoms:
        return [ atoms ]

def set_relaxation_radius(supercell, site_idxs, R=2.0):
    """ Adds constraints so that relaxation only occurs within R*min_dist of sites"""
    relaxation_idxs = set()
    
    for site_idx in site_idxs:
        # select relaxation radius to be R * minimum interatomic distance
        site_dists = supercell.get_distances(site_idx,True)
        
        # set radius in terms of nearest neighbor (if exists) or lattice constant:
        neighbor_dists = [ d for d in site_dists if d > 0.0 ]
        if len(neighbor_dists) > 0:
            relaxation_radius = R*min(neighbor_dists)
        else:
            relaxation_radius = R*min(supercell.cell.lengths())
        
        # find neighbors within relaxation radius of site:
        nl = NeighborList([relaxation_radius]*len(supercell))
        nl.update(supercell)
        idxs, offsets = nl.get_neighbors(site_idx)
        relaxation_idxs |= set(idxs)

    # add constraint to supercell:    
    constraint = FixAtoms(indices=[ 
        a.index for a in supercell 
        if a.index not in relaxation_idxs
    ])
    supercell.set_constraint(constraint)


def generate_supercells(pmap_args):
    formula, (tokens, base_tokens, atoms, known_dir, defect_dir) = pmap_args
    
    # check if supercells have already been generated:
    cached_supercells = glob.glob(os.path.join(KNOWN_STRUCTURE_EXPORT_DIR,f'{formula}__*.poscar')) + \
                        glob.glob(os.path.join(DEFECT_STRUCTURE_EXPORT_DIR,f'{formula}__*.poscar'))
    if len(cached_supercells) > 0:
        return    
 
    supercells = generate_defect_supercells(tokens, 
                                            base_tokens, 
                                            atoms,
                                            random=True,
                                            max_n_samples=10,
                                            formula=formula)
    if len(supercells) <= 0:
        print(f'{formula}: No supercell found, supercell is too complex.')
        return
    
    # write atoms object as a .traj file 
    # (.traj is ASE's native format, which should include constraints, etc.)
    for i, sc in enumerate(supercells):
        export_filename = f'{formula}__{i+1:02d}.traj'
        if sc == atoms:
            export_path = os.path.join(known_dir, export_filename)
            print(f'{formula}: Using base unit cell. (size: {len(sc)})')
            sc.write(export_path, format='traj')
            return
        else:
            export_path = os.path.join(defect_dir, export_filename)
            print(f'{formula}: Exported {i+1}/{len(supercells)} supercells. (size: {len(sc)})')
            sc.write(export_path, format='traj')


def main():
   
    args = parser.parse_args()
 
    # read periodic table CSV and parse elements:
    ptable_df = pd.read_csv(args.ptable_csv)
    elements = list(ptable_df['Symbol'])
    formula_re = build_formula_regex(elements)

    # import unitcell data:
    unitcell_files = {}
    for filename in os.listdir(args.unitcell_data_dir):
        formula = filename.split('__')[0]

        if formula not in unitcell_files:
            unitcell_files[formula] = []

        unitcell_files[formula].append(filename)

    base_unitcell_data = {}

    print('Importing base unit cells...')
    for formula in tqdm(unitcell_files):
        tokens = parse_formula_tokens(formula, formula_re)
        unmixed_tokens = get_unmixed_material_tokens(tokens)
        
        # determine which elements are defects:
        for base_tokens in unmixed_tokens:
            base_formula = formula_from_tokens(base_tokens)
            base_filename = f'{formula}__{base_formula}.poscar'
             
            # search for best structural match to add defects to:
            if base_filename in unitcell_files[formula]:
                
                # Import best match as an Atoms instance:
                base_unitcell_path = os.path.join(args.unitcell_data_dir, base_filename)
                atoms = read(base_unitcell_path)
                base_unitcell_data[formula] = (tokens, base_tokens, atoms, 
                                               args.known_structure_export_dir, 
                                               args.defect_structure_export_dir)
                break
        else:
            print('Unable to find match for: ', formula)

    print('Solving for optimal fcc-like supercells ...')
    
    with Pool(processes=args.processes) as P:
        P.map(generate_supercells, list(base_unitcell_data.items()))

if __name__ == '__main__':
    main()
