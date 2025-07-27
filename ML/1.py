from pymatgen.io.cif import CifParser
from pymatgen.core.structure import Structure
from pymatgen.ext.matproj import MPRester
import numpy as np
import pandas as pd
from ase.io import read

import os


API_KEY = "aNUnHzxy70Ms9b7xKdMLPaqfHCkUl5UA"


cif_files = ["4124511.cif", "1540301.cif"]
material_ids = [
    "mp-23", "mp-27", "mp-19009", "mp-62", "mp-1572", "mp-13", "mp-101",
    "mp-134", "mp-2652", "mp-153", "mp-58", "mp-20194", "mp-30", "mp-78",
    "mp-81", "mp-131", "mp-88", "mp-23265", "mp-136", "mp-1176", "mp-1518",
    "mp-1538", "mp-127", "mp-60", "mp-135", "mp-1143", "mp-87", "mp-1540",
    "mp-1520", "mp-2657", "mp-7000", "mp-2858", "mp-1265", "mp-117",
    "mp-19770", "mp-19399", "mp-19306", "mp-18759", "mp-2133", "mp-540806",
    "mp-19207", "mp-540795", "mp-540798"
]


mpr = MPRester(API_KEY)


def estimate_density_empirical(structure):
    return 0.1 * len(structure)

def calculate_density_dft(structure):
    total_mass = np.sum([site.species.weight for site in structure])
    volume = structure.volume
    return total_mass / volume if volume > 0 else np.nan

def calculate_density_packing(structure):
    volume = structure.volume
    if volume == 0:
        return np.nan
    packing_density = len(structure) / volume
    return packing_density * sum([site.species.weight for site in structure])

def predict_density_ml(structure):
    return 0.05 * len(structure)

def calculate_density_ase(cif_file):
    try:
        atoms = read(cif_file)
        volume = atoms.get_volume()
        mass = atoms.get_masses().sum()
        return (mass / 6.022e23) / (volume * 1e-24) if volume > 0 else np.nan
    except Exception as e:
        print(f"Error processing {cif_file} with ASE: {e}")
        return np.nan


def predict_formation_energy(structure):
    try:
        gen = formation_energy.prepare_data(data_input=[structure], input_format='cif')
        model = formation_energy.FormationEnergyPredictor()
        yp = model.predict(gen, return_all_ensembles=True)
        return yp[0] if isinstance(yp, list) else yp
    except Exception as e:
        print(f"Error predicting formation energy: {e}")
        return np.nan


def get_mp_data(identifier, identifier_type='material_id'):
    try:
        if identifier_type == 'material_id':
            summary = mpr.summary.get_data_by_id(identifier)
            return {
                'Formula': summary.formula_pretty,
                'Formation_Energy (eV/atom)': summary.formation_energy_per_atom,
                'Density (g/cm³)': summary.density
            }
        elif identifier_type == 'formula':
            results = mpr.summary.search(formula=identifier)
            if results:
                return {
                    'Formula': results[0].formula_pretty,
                    'Formation_Energy (eV/atom)': results[0].formation_energy_per_atom,
                    'Density (g/cm³)': results[0].density
                }
        return None
    except Exception as e:
        print(f"Error fetching MP data for {identifier}: {e}")
        return None


cif_data = []
for cif_file in cif_files:
    if not os.path.exists(cif_file):
        print(f"CIF file {cif_file} not found.")
        continue
    try:
        cif_parser = CifParser(cif_file)
        structure = cif_parser.get_structures()[0]
        formula = structure.composition.reduced_formula
        
        cif_data.append({
            'Identifier': cif_file,
            'Formula': formula,
            'Empirical_Density (g/cm³)': estimate_density_empirical(structure),
            'DFT_Density (g/cm³)': calculate_density_dft(structure),
            'Packing_Density (g/cm³)': calculate_density_packing(structure),
            'ML_Density (g/cm³)': predict_density_ml(structure),
            'ASE_Density (g/cm³)': calculate_density_ase(cif_file),
            'Formation_Energy (eV/atom)': predict_formation_energy(structure)
        })
    except Exception as e:
        print(f"Error processing CIF {cif_file}: {e}")


mp_data = []
for material_id in material_ids:
    mp_result = get_mp_data(material_id, 'material_id')
    if mp_result:
        mp_data.append({
            'Identifier': material_id,
            'Formula': mp_result['Formula'],
            'Formation_Energy (eV/atom)': mp_result['Formation_Energy (eV/atom)'],
            'Density (g/cm³)': mp_result['Density (g/cm³)']
        })


all_data = cif_data + mp_data
df = pd.DataFrame(all_data)


print("\nPredicted and Retrieved Material Properties:")
print(df)


excel_file = 'ml-dataset.xlsx'
if os.path.exists(excel_file):
    excel_data = pd.read_excel(excel_file, sheet_name="Table")
    column_name = 'Active component type'
    if column_name in excel_data.columns:
        unique_count = excel_data[column_name].nunique()
        unique_names = excel_data[column_name].unique()
        print(f"\nNumber of unique names in '{column_name}': {unique_count}")
        print("Unique names:")
        print(unique_names)
    else:
        print(f"\nError: Column '{column_name}' does not exist in the dataset.")
else:
    print(f"\nExcel file {excel_file} not found.")