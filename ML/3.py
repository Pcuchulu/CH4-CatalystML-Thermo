import pandas as pd
import numpy as np



density_formation_data = {
    'Ni': {'formation_energy': 0.046, 'density': 9.22},
    'Co': {'formation_energy': 0.025, 'density': 9.2},
    'ReOx': {'formation_energy': 0.004, 'density': 20.87},
    'Ru': {'formation_energy': 0.131, 'density': 12.32},
    'NiO': {'formation_energy': -1.218, 'density': 6.76},
    'Ni-SiO2': {'formation_energy': -2.080852, 'density': 3.2926},
    'Rh': {'formation_energy': 0.022, 'density': 12.38},
    'Ce': {'formation_energy': 0.075, 'density': 8.98},
    'Fe': {'formation_energy': 0.148, 'density': 7.59},
    'Pd': {'formation_energy': 0.015, 'density': 11.73},
    'Mg': {'formation_energy': 0.04, 'density': 1.81},
    'Y2O3': {'formation_energy': -3.905, 'density': 5.03},
    'Ba': {'formation_energy': 0.175, 'density': 3.62},
    'K': {'formation_energy': 0.015, 'density': 0.83},
    'CeO2': {'formation_energy': -3.755, 'density': 6.99},
    'Cu': {'formation_energy': 0.033, 'density': 9.2},
    'Ca': {'formation_energy': 0.18, 'density': 1.44},
    'Mn': {'formation_energy': 0.421, 'density': 8.74},
    'Zr': {'formation_energy': 0.057, 'density': 6.52},
    'Y': {'formation_energy': 0.098, 'density': 4.44},  
    'La2O3': {'formation_energy': 3.733, 'density': 5.9},
    'VOx': {'formation_energy': 0.247, 'density': 6.2},
    'La': {'formation_energy': 0.019, 'density': 6.12},
    'Pr': {'formation_energy': 0.126, 'density': 6.34},
    'Eu': {'formation_energy': 0.041, 'density': 5.58},
    'Gd': {'formation_energy': 0.15, 'density': 7.75},
    'Na': {'formation_energy': 0.016, 'density': 1.02},
    'Cs': {'formation_energy': 0.019, 'density': 1.8},
    'Li': {'formation_energy': 0.002, 'density': 0.57},
    'Al2O3': {'formation_energy': -3.352, 'density': 3.87},
    'Sr': {'formation_energy': 0.045, 'density': 2.61},
    'Yb': {'formation_energy': 0.014, 'density': 7.01},
    'Sm': {'formation_energy': 0.019, 'density': 7.27},
    'TiO2': {'formation_energy': -3.316, 'density': 3.86},
    'SiO2': {'formation_energy': -3.04, 'density': 2.63},
    'ZrO2': {'formation_energy': -3.755, 'density': 6.04},
    'SBA-15': {'formation_energy': -3.02, 'density': 1.97},
    'MgO': {'formation_energy': -2.995, 'density': 3.63},
    'SiC': {'formation_energy': -0.204, 'density': 3.23},
    'AC': {'formation_energy': 0.009, 'density': 0.9},
    'MSN': {'formation_energy': 0.61, 'density': 2.31},
    'MCM-41': {'formation_energy': -3.04, 'density': 2.2},
    'Zeolite': {'formation_energy': -3.081, 'density': 1.561428571},
    'NC': {'formation_energy': 0.008, 'density': 1.94},
    'Na-Zeolite': {'formation_energy': -3.081, 'density': 2.31},
    'Cs-Zeolite': {'formation_energy': -3.081, 'density': 1.95},
    'Fe2O3': {'formation_energy': -1.588, 'density': 4.56},
    'OCF': {'formation_energy': 0.008, 'density': 1.94},
    'AX': {'formation_energy': -2.349758, 'density': 2.053934627},
    'B': {'formation_energy': 0.02, 'density': 2.32},
    'PC': {'formation_energy': -0.51, 'density': 3.367},
    'eg-C3N4': {'formation_energy': 0.322, 'density': 3.5},
    'Cr2O3': {'formation_energy': -2.367, 'density': 5.17},
    'YMnO3': {'formation_energy': -0.3008, 'density': 5.09},
    'Mn3O4': {'formation_energy': -2.05, 'density': 4.89},
    'ZnO': {'formation_energy': -1.635, 'density': 5.71},
    'Gd2O3': {'formation_energy': -3.718, 'density': 8.48},
    'Sm2O3': {'formation_energy': -3.753, 'density': 7.08},
    'Pr2O3': {'formation_energy': -3.603, 'density': 6.9},
    'MgO-Nd2O3': {'formation_energy': -8.83, 'density': 4.31},
    'Nd2O3': {'formation_energy': -3.665, 'density': 6.49}
}


prep_method_data = {
    'MC': {'scalability': 4, 'cost': 2},
    'WI': {'scalability': 5, 'cost': 1},
    'FM': {'scalability': 5, 'cost': 1},
    'IWI': {'scalability': 5, 'cost': 1},
    'CP': {'scalability': 4, 'cost': 3},
    'SGP': {'scalability': 4, 'cost': 3},
    'AE': {'scalability': 3, 'cost': 2},
    'SCT': {'scalability': 4, 'cost': 2},
    'IWI-DBD': {'scalability': 3, 'cost': 5},
    'DPA': {'scalability': 3, 'cost': 3},
    'DPU': {'scalability': 4, 'cost': 3},
    'Commercial': {'scalability': 5, 'cost': 1},
    'FSP': {'scalability': 3, 'cost': 5},
    'DBD': {'scalability': 3, 'cost': 5},
    'CI': {'scalability': 4, 'cost': 3},
    'EISA': {'scalability': 2, 'cost': 5},
    'OUCI': {'scalability': 3, 'cost': 3},
    'DP': {'scalability': 3, 'cost': 3},
    'CC': {'scalability': 4, 'cost': 3},
    'ME': {'scalability': 5, 'cost': 1},
    'CP-MSM': {'scalability': 3, 'cost': 3},
    'UH': {'scalability': 3, 'cost': 3},
    'H': {'scalability': 4, 'cost': 3},
    'OH': {'scalability': 3, 'cost': 3},
    'MI': {'scalability': 4, 'cost': 3}
}


excel_file = 'ml-dataset.xlsx'  
sheet1 = pd.read_excel(excel_file, sheet_name='table coded')
sheet2 = pd.read_excel(excel_file, sheet_name='Dataset')

sheet1 = pd.read_excel(excel_file, sheet_name='table coded')
print("Sheet1 columns before renaming:", sheet1.columns.tolist())


unified_columns = [
    'Number of data', 'Reference', 'Experiment No', 'Train/Test Set',
    'Active component type', 'Active component type density', 'Active component type formation energy', 'Active component content (wt%)',
    'Promoter type', 'Promoter type density', 'Promoter type formation energy', 'Promoter content (wt%)',
    'Support a type', 'Support a type density', 'Support a type formation energy', 'Support a content (wt%)',
    'Support b type', 'Support b type density', 'Support b type formation energy', 'Support b content (wt%)',
    'Preparation method', 'Preparation Scalability', 'Preparation cost',
    'Calcination Temperature (C)', 'Calcination time (h)',
    'Reduction Temperature (C)', 'Reduction Pressure (bar)', 'Reduction time (h)', 'Reduced hydrogen content (vol%)',
    'Temperature (C)', 'Pressure (bar)', 'Weight hourly space velocity [mgcat/(min·ml)]', 'Time on Stream (h)',
    'Content of inert components in raw materials (vol%)', 'CO % in feed', 'CH4 % in feed', 'H2O % in feed',
    'H2/CO2 ratio (mol/mol)', 'CO2 conversion ratio(%)', 'CH4 selectivity (%)', 'CH4 yield(%)'
]


sheet1_column_mapping = {
    'Active component type': 'Active component type',
    'Active component type density': 'Active component type density',
    'Active component type formation energy': 'Active component type formation energy',
    'Active component content (wt%)': 'Active component content (wt%)',
    'Promoter type': 'Promoter type',
    'Promoter type density': 'Promoter type density',
    'Promoter type formation energy': 'Promoter type formation energy',
    'Promoter content (wt%)': 'Promoter content (wt%)',
    'Support\na type': 'Support a type',
    'Support\na type density': 'Support a type density',
    'Support\na type formation energy': 'Support a type formation energy',
    'Support a\ncontent (wt%)': 'Support a content (wt%)',
    'Support\nb type': 'Support b type',
    'Support\nb type density': 'Support b type density',
    'Support\nb type formation energy': 'Support b type formation energy',
    'Preparation method': 'Preparation method',
    'Preparation Scalability': 'Preparation Scalability',
    'Preparation cost': 'Preparation cost',
    'Calcination Temperature (℃)': 'Calcination Temperature (C)',
    'Calcination time (h)': 'Calcination time (h)',
    'Reduction Temperature (℃)': 'Reduction Temperature (C)',
    'Reduction Pressure (bar)': 'Reduction Pressure (bar)',
    'Reduction time  (h)': 'Reduction time (h)',
    'Reduced hydrogen content (vol%)': 'Reduced hydrogen content (vol%)',
    'Temperature (℃)': 'Temperature (C)',
    'Pressure (bar)': 'Pressure (bar)',
    'Weight hourly space velocity [mgcat/(min·ml)]': 'Weight hourly space velocity [mgcat/(min·ml)]',
    'Content of inert components in raw materials (vol%)': 'Content of inert components in raw materials (vol%)',
    'H2/CO2\nratio (mol/mol)': 'H2/CO2 ratio (mol/mol)',
    'CO2\nconversion ratio(%)': 'CO2 conversion ratio(%)',
    'CH4\nselectivity (%)': 'CH4 selectivity (%)',
    'CH4\nyield(%)': 'CH4 yield(%)',
    'Reference': 'Reference'
}


sheet2_column_mapping = {
    'Number of data': 'Number of data',
    'Reference': 'Reference',
    'Experiment No': 'Experiment No',
    'Train/Test Set': 'Train/Test Set',
    'Base': 'Active component type',
    'Base wt%': 'Active component content (wt%)',
    'Base 2': 'Promoter type',
    'Base 2 wt%': 'Promoter content (wt%)',
    'Support': 'Support a type',
    'Support wt%': 'Support a content (wt%)',
    'Support 2': 'Support b type',
    'Catalyst preparation method': 'Preparation method',
    'Calcination Temperature (C)': 'Calcination Temperature (C)',
    'Calcination time (h)': 'Calcination time (h)',
    'Reduction Temperature (C)': 'Reduction Temperature (C)',
    'Reduction Pressure (bar)': 'Reduction Pressure (bar)',
    'Reduction time  (h)': 'Reduction time (h)',
    'Reduction  H2 %': 'Reduced hydrogen content (vol%)',
    'Temperature (C)': 'Temperature (C)',
    'Pressure (bar)': 'Pressure (bar)',
    'W/F (mgcat/minml)': 'Weight hourly space velocity [mgcat/(min·ml)]',
    'Time on Stream (h)': 'Time on Stream (h)',
    'CO % in feed': 'CO % in feed',
    'Inert % in feed': 'Content of inert components in raw materials (vol%)',
    'CH4 % in feed': 'CH4 % in feed',
    'H2O % in feed': 'H2O % in feed',
    'H2/CO2 in feed': 'H2/CO2 ratio (mol/mol)',
    'CO2 Conversion %': 'CO2 conversion ratio(%)'
}


sheet1 = sheet1.rename(columns=sheet1_column_mapping)
sheet2 = sheet2.rename(columns=sheet2_column_mapping)


def update_density_formation(df, type_col, density_col, energy_col):
    df[density_col] = df[type_col].map(lambda x: density_formation_data.get(str(x).strip(), {'density': 0})['density'] if pd.notna(x) else 0)
    df[energy_col] = df[type_col].map(lambda x: density_formation_data.get(str(x).strip(), {'formation_energy': 0})['formation_energy'] if pd.notna(x) else 0)
    return df


sheet1 = update_density_formation(sheet1, 'Active component type', 'Active component type density', 'Active component type formation energy')
sheet1 = update_density_formation(sheet1, 'Promoter type', 'Promoter type density', 'Promoter type formation energy')
sheet1 = update_density_formation(sheet1, 'Support a type', 'Support a type density', 'Support a type formation energy')
sheet1 = update_density_formation(sheet1, 'Support b type', 'Support b type density', 'Support b type formation energy')


sheet2 = update_density_formation(sheet2, 'Active component type', 'Active component type density', 'Active component type formation energy')
sheet2 = update_density_formation(sheet2, 'Promoter type', 'Promoter type density', 'Promoter type formation energy')
sheet2 = update_density_formation(sheet2, 'Support a type', 'Support a type density', 'Support a type formation energy')
sheet2 = update_density_formation(sheet2, 'Support b type', 'Support b type density', 'Support b type formation energy')


def assign_scalability_cost(df):
    df['Preparation Scalability'] = df['Preparation method'].map(lambda x: prep_method_data.get(str(x).strip(), {'scalability': ''})['scalability'] if pd.notna(x) else '')
    df['Preparation cost'] = df['Preparation method'].map(lambda x: prep_method_data.get(str(x).strip(), {'cost': ''})['cost'] if pd.notna(x) else '')
    return df

sheet1 = assign_scalability_cost(sheet1)
sheet2 = assign_scalability_cost(sheet2)


sheet1['Reference'] = sheet1['Reference'].apply(lambda x: f"1-[{x}]" if pd.notna(x) else '')
sheet2['Reference'] = sheet2['Reference'].apply(lambda x: f"2-[{x}]" if pd.notna(x) else '')


sheet1_unified = pd.DataFrame(columns=unified_columns)
sheet2_unified = pd.DataFrame(columns=unified_columns)


for col in unified_columns:
    if col in sheet1.columns:
        sheet1_unified[col] = sheet1[col]
    if col in sheet2.columns:
        sheet2_unified[col] = sheet2[col]


merged_df = pd.concat([sheet1_unified, sheet2_unified], ignore_index=True)


for col in merged_df.columns:
    if merged_df[col].dtype in ['float64', 'int64']:
        merged_df[col] = merged_df[col].fillna(0)
    else:
        merged_df[col] = merged_df[col].fillna('')


merged_df.to_csv('merged_data.csv', index=False)
print("Merged data saved to 'merged_data.csv'")