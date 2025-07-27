import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split
import os
import warnings
warnings.filterwarnings('ignore')


plt.rcParams['text.usetex'] = False


try:
    import scienceplots
    plt.style.use(['science', 'grid'])
except ImportError:
    print("Warning: 'scienceplots' not installed. Falling back to 'ggplot' style.")
    plt.style.use('ggplot')


features = [
    'Active component formation energy',
    'Active component density',
    'Active component content (wt percent)',
    'Promoter formation energy',
    'Promoter density',
    'Promoter content (wt percent)',
    'Support a formation energy',
    'Support a density',
    'Support a content (wt percent)',
    'Support b formation energy',
    'Support b density',
    'Calcination Temperature (C)',
    'Calcination time (h)',
    'Reduction Temperature (C)',
    'Reduction Pressure (bar)',
    'Reduction time (h)',
    'Reduced hydrogen content (vol percent)',
    'Temperature (C)',
    'Pressure (bar)',
    'Weight hourly space velocity [mgcat/(min·ml)]',
    'Content of inert components in raw materials (vol percent)',
    'h2/co2 ratio (mol/mol)',
    'Preparation Scalability',
    'Preparation cost'
]

abbreviations = {
    'Active component density': 'ACD',
    'Active component formation energy': 'ACFE',
    'Active component content (wt percent)': 'ACW',
    'Promoter density': 'PD',
    'Promoter formation energy': 'PFE',
    'Promoter content (wt percent)': 'PW',
    'Support a density': 'SAD',
    'Support a formation energy': 'SAFE',
    'Support a content (wt percent)': 'SAW',
    'Support b density': 'SBD',
    'Support b formation energy': 'SBFE',
    'Calcination Temperature (C)': 'CTC',
    'Calcination time (h)': 'CTH',
    'Reduction Temperature (C)': 'RTC',
    'Reduction Pressure (bar)': 'RPB',
    'Reduction time (h)': 'RTH',
    'Reduced hydrogen content (vol percent)': 'RHC',
    'Temperature (C)': 'T',
    'Pressure (bar)': 'P',
    'Weight hourly space velocity [mgcat/(min·ml)]': 'WHSV',
    'Content of inert components in raw materials (vol percent)': 'CIRMW',
    'h2/co2 ratio (mol/mol)': 'H2/CO2',
    'Preparation Scalability': 'PS',
    'Preparation cost': 'PC'
}


def load_data(file_path):
    try:
        if file_path.endswith('.csv'):
            data = pd.read_csv(file_path, encoding='utf-8')
        elif file_path.endswith('.xlsx'):
            data = pd.read_excel(file_path, sheet_name="Table")
    except UnicodeDecodeError:
        data = pd.read_csv(file_path, encoding='ISO-8859-1')
    except Exception as e:
        print(f"Error loading file: {e}")
        raise
    return data


file_path = 'ml2.csv'
data = load_data(file_path)


data = data.replace(['?', 'NA', 'nan'], np.nan)
numeric_columns = [col for col in data.columns if col != 'Preparation method']
data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, errors='coerce')
data['Preparation method'] = data['Preparation method'].astype('category')


imputer = IterativeImputer(random_state=42)
data['co2 conversion ratio (percent)'] = imputer.fit_transform(data[['co2 conversion ratio (percent)']])


print(f"Columns in data: {list(data.columns)}")
print(f"Missing values in key columns:\n{data[['Preparation cost', 'Preparation Scalability', 'Preparation method', 'co2 conversion ratio (percent)']].isna().sum()}")
print(f"Unique values in Preparation method:\n{data['Preparation method'].unique()}")
print(f"Sample of key columns:\n{data[['Preparation cost', 'Preparation Scalability', 'Preparation method', 'co2 conversion ratio (percent)']].head()}")


try:
    X = data[features]
except KeyError as e:
    print(f"Error: One or more feature columns not found in dataset: {e}")
    print(f"Available columns: {list(data.columns)}")
    raise
X.columns = [abbreviations.get(col, col) for col in X.columns]


targets = {
    'CO2 conversion ratio (percent)': data['co2 conversion ratio (percent)'],
    'CH4 selectivity (percent)': data['ch4 selectivity (percent)'],
    'CH4 yield (percent)': data['ch4 yield (percent)']
}


target_abbr = {
    'co2 conversion ratio (percent)': 'CO2_CR',
    'ch4 selectivity (percent)': 'CH4_S',
    'ch4 yield (percent)': 'CH4_Y'
}
for target_name, y in targets.items():
    y.name = target_abbr.get(y.name, y.name)


os.makedirs('plots', exist_ok=True)


def generate_heatmaps(X, targets, abbreviations):
    for target_name, y in targets.items():
        X_train, _, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        corrmat = X_train.corr(method='pearson')
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            corrmat,
            cmap="RdBu",
            annot=False,
            linewidths=0.5,
            ax=ax,
            xticklabels=X.columns,
            yticklabels=X.columns
        )
        plt.title(rf'{target_name.replace(" (percent)", " (\%)")} - Pearson Correlation Coefficient')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
        plt.close()


def data_distribution_analysis(X, y, target_name, data):
    
    plt.rcParams.update({
        "text.usetex": False,  
        "font.family": "Arial",  
    })

    
    if target_name == 'co2 conversion ratio (percent)':
        display_name = 'CO$_2$ conversion ratio (%)'
        title_name = 'CO$_2$ Conversion Ratio (%)'
    else:
        display_name = target_name.replace('percent', '%')
        title_name = target_name.replace('percent', '%')

    
    selected_features = ['ACFE', 'ACD', 'T', 'P', target_name]
    y_df = pd.DataFrame(y, columns=[target_name])
    pairplot_data = pd.concat([X, y_df], axis=1)[selected_features]
    
    g = sns.pairplot(pairplot_data, diag_kind='kde')
    
    for ax in g.axes.flatten():
        if ax.get_ylabel() == target_name:
            ax.set_ylabel(display_name, fontsize=12)
        if ax.get_xlabel() == target_name:
            ax.set_xlabel(display_name, fontsize=12)
    plt.savefig(f'plots/pairplot_{target_name}.png', dpi=600, bbox_inches='tight')
    plt.close()

    
    plt.figure(figsize=(8, 6))
    sns.violinplot(x=pd.cut(X['T'], bins=5), y=y, palette='viridis')
    plt.xlabel('Temperature (°C) Bins', fontsize=12, weight='bold')
    plt.ylabel(display_name, fontsize=12, weight='bold')
    plt.title(title_name + ' Distribution Across Temperature Bins')
    plt.show()
    plt.close()

    
    plt.figure(figsize=(8, 6))
    sns.violinplot(x=pd.cut(X['P'], bins=5), y=y, palette='viridis')
    plt.xlabel('Pressure (bar) Bins', fontsize=12, weight='bold')
    plt.ylabel(display_name, fontsize=12, weight='bold')
    plt.title(title_name + ' Distribution Across Pressure Bins')
    plt.show()
    plt.close()

    
    if target_name == 'co2 conversion ratio (percent)':
        data['Unique Count'] = data['Active component formation energy'].map(
            data['Active component formation energy'].value_counts())
        
        
        fig, ax = plt.subplots(figsize=(6, 5))
        sc = sns.scatterplot(
            data=data,
            x='Temperature (C)',
            y=target_name,
            hue='Active component formation energy',
            size='Unique Count',
            sizes=(20, 200),
            palette='rainbow',
            legend=False,
            ax=ax,
            alpha=0.5
        )
        ax.set_xlabel('Temperature (°C)', fontsize=12, weight='bold')
        ax.set_ylabel(display_name, fontsize=12, weight='bold')
        plt.show()
        plt.close()
        

generate_heatmaps(X, targets, abbreviations)
for target_name, y in targets.items():
    print(f"\nProcessing data distribution for {target_name}...")
    data_distribution_analysis(X, y, target_name, data)