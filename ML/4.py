import pandas as pd
import json


csv_file = 'C:\\Users\\pcu\\Desktop\\merged_data.csv'  
json_file = 'C:\\Users\\pcu\\Desktop\\adsorption_data.json'  
output_csv = 'C:\\Users\\pcu\\Desktop\\merged_data_with_adsorption.csv'  


df = pd.read_csv(csv_file)


with open(json_file, 'r') as f:
    json_data = json.load(f)


adsorption_energies = {}
for result in json_data.get('results', []):
    adsorption_measurement = result.get('adsorption_measurement', {})
    adsorbate_formula = adsorption_measurement.get('adsorbate_species', {}).get('formula', '')
    is_most_stable_site = adsorption_measurement.get('is_most_stable_site', False)
    
    
    if adsorbate_formula == 'CO' and is_most_stable_site:
        material_name = adsorption_measurement.get('bulk_surface_property_set', {}).get('bulk_surface_material', {}).get('name', '')
        adsorption_energy = adsorption_measurement.get('adsorption_energy', 0)
        
        
        if material_name and material_name not in adsorption_energies:
            adsorption_energies[material_name] = adsorption_energy


component_types = [
    ('Active component type', 'Active component adsorption energy'),
    ('Promoter type', 'Promoter adsorption energy'),
    ('Support a type', 'Support a type adsorption energy'),
    ('Support b type', 'Support b type adsorption energy')
]


for type_col, energy_col in component_types:
    if type_col in df.columns:
        
        col_index = df.columns.get_loc(type_col)
        
        energy_values = df[type_col].map(lambda x: adsorption_energies.get(str(x).strip(), 0) if pd.notna(x) else 0)
        
        df.insert(col_index + 1, energy_col, energy_values)
    else:
        print(f"Warning: Column '{type_col}' not found in CSV. Adding '{energy_col}' at the end.")
        df[energy_col] = 0


df.to_csv(output_csv, index=False)
print(f"Updated CSV saved to '{output_csv}'")