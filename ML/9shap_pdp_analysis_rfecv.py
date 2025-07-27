import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import FuncFormatter
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, GradientBoostingRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.inspection import partial_dependence
import joblib
import os
import shap
import warnings
warnings.filterwarnings('ignore')


try:
    import scienceplots
    plt.style.use(['science', 'no-latex', 'grid'])
except ImportError:
    print("Warning: 'scienceplots' not installed. Falling back to 'seaborn' style.")
    plt.style.use('seaborn')
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 12,
    'figure.titlesize': 18
})


FEATURE_MAP = {
    'Active component formation energy': 'ACFE',
    'Active component density': 'ACD',
    'Active component content (wt percent)': 'ACW',
    'Promoter formation energy': 'PFE',
    'Promoter density': 'PD',
    'Promoter content (wt percent)': 'PW',
    'Support a formation energy': 'SAFE',
    'Support a density': 'SAD',
    'Support a content (wt percent)': 'SAW',
    'Support b formation energy': 'SBFE',
    'Support b density': 'SBD',
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


MODEL_ABBREVIATIONS = {
    'Random Forest': 'RF',
    'Bagging Regressor': 'BR',
    'Gradient Boosting Decision Tree': 'GBDT',
    'LightGBM': 'LGBM',
    'XGBoost': 'XGB',
    'CatBoost': 'CB'
}


pdp_feature_pairs = [
    ('Active component formation energy', 'Active component density'),
    ('Promoter formation energy', 'Promoter density'),
    ('Support a formation energy', 'Support a density'),
    ('Temperature (C)', 'h2/co2 ratio (mol/mol)')
]


TARGETS = {
    'CO2_Conversion': 'co2 conversion ratio (percent)',
    'CH4_Selectivity': 'ch4 selectivity (percent)',
    'CH4_Yield': 'ch4 yield (percent)'
}


TARGET_ABBREVIATIONS = {
    'co2 conversion ratio (percent)': 'CO2_CR',
    'ch4 selectivity (percent)': 'CH4_S',
    'ch4 yield (percent)': 'CH4_Y'
}

def load_data(file_path):
    """Load dataset with appropriate encoding"""
    print("Attempting to load data...")
    try:
        if file_path.endswith('.csv'):
            data = pd.read_csv(file_path, encoding='utf-8')
        elif file_path.endswith('.xlsx'):
            data = pd.read_excel(file_path, sheet_name="Table")
        print(f"Data loaded successfully. Shape: {data.shape}, Columns: {list(data.columns)}")
        return data
    except UnicodeDecodeError:
        data = pd.read_csv(file_path, encoding='ISO-8859-1')
        print(f"Data loaded successfully with ISO-8859-1 encoding. Shape: {data.shape}, Columns: {list(data.columns)}")
        return data
    except Exception as e:
        print(f"Error loading file: {e}")
        raise

def load_selected_features(target_name, model_name):
    """Load selected features for target and model from saved_features directory"""
    model_abbr = MODEL_ABBREVIATIONS.get(model_name, model_name.replace(' ', '_'))
    feature_file = os.path.join('saved_features', f"{target_name}_{model_abbr}_selected_features.txt")
    if os.path.exists(feature_file):
        try:
            with open(feature_file, 'r') as f:
                features = [line.strip() for line in f if line.strip()]
            if not features:
                print(f"Error: Feature file {feature_file} is empty")
                return None
            reverse_feature_map = {v: k for k, v in FEATURE_MAP.items()}
            mapped_features = []
            for feature in features:
                if feature in reverse_feature_map:
                    mapped_features.append(reverse_feature_map[feature])
                else:
                    print(f"Warning: Feature '{feature}' in {feature_file} not found in FEATURE_MAP")
            if not mapped_features:
                print(f"Error: No valid features mapped for {feature_file}")
                return None
            print(f"Loaded {len(features)} features from {feature_file}, mapped to {len(mapped_features)} valid features: {mapped_features}")
            return mapped_features
        except Exception as e:
            print(f"Error loading features from {feature_file}: {e}")
            return None
    else:
        print(f"Error: Feature file {feature_file} not found")
        return None

def preprocess_data(data, selected_features):
    """Preprocess the dataset for selected features"""
    print("Preprocessing data...")
    try:
        data = data.copy()
        data = data.replace(['?', 'NA', 'nan'], np.nan)
        numeric_columns = [col for col in selected_features if col in data.columns and col != 'Preparation method']
        data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, errors='coerce')
        if 'Preparation method' in data.columns and 'Preparation method' in selected_features:
            data['Preparation method'] = data['Preparation method'].astype('category')
        return data
    except Exception as e:
        print(f"Error preprocessing data: {e}")
        return None

def get_feature_abbreviations(features):
    """Map full feature names to abbreviations"""
    return [FEATURE_MAP.get(f, f) for f in features]

def plot_3d_pdp_matplotlib(X_train_scaled, models, X_columns, target_name, feature_pairs):
    """Generate 3D Partial Dependence Plots using Matplotlib, saved as PNG"""
    for model_name, model in models.items():
        for feature_1, feature_2 in feature_pairs:
            try:
                print(f"Computing PDP for {model_name} - {target_name} ({feature_1} vs {feature_2})")
                feature_1_idx = X_columns.get_loc(feature_1)
                feature_2_idx = X_columns.get_loc(feature_2)
                pdp_result = partial_dependence(
                    model, X_train_scaled, features=[feature_1_idx, feature_2_idx],
                    kind="average", grid_resolution=5
                )
                XX, YY = np.meshgrid(pdp_result["grid_values"][0], pdp_result["grid_values"][1])
                Z = pdp_result["average"][0].T

                print(f"Rendering PDP plot for {model_name} - {target_name} ({feature_1} vs {feature_2})")
                fig = plt.figure(figsize=(6, 5))
                ax = fig.add_subplot(111, projection='3d')
                surf = ax.plot_surface(XX, YY, Z, cmap='viridis', edgecolor='none')
                ax.set_xlabel(FEATURE_MAP.get(feature_1, feature_1), fontsize=12, labelpad=10)
                ax.set_ylabel(FEATURE_MAP.get(feature_2, feature_2), fontsize=12, labelpad=10)
                ax.set_zlabel("Partial Dependence", fontsize=12, labelpad=10)
                formatter = FuncFormatter(lambda x, _: f'{x:.2f}')
                ax.xaxis.set_major_formatter(formatter)
                ax.yaxis.set_major_formatter(formatter)
                ax.zaxis.set_major_formatter(formatter)
                ax.tick_params(axis='both', which='major', labelsize=10)
                ax.tick_params(axis='z', which='major', labelsize=10)
                fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1)
                plt.title(f'PDP: {MODEL_ABBREVIATIONS.get(model_name, model_name)} - {TARGET_ABBREVIATIONS.get(TARGETS[target_name], target_name)}', fontsize=14, pad=20)
                plt.tight_layout()
                safe_feature_1 = FEATURE_MAP.get(feature_1, feature_1).replace('/', '_')
                safe_feature_2 = FEATURE_MAP.get(feature_2, feature_2).replace('/', '_')
                safe_model_name = MODEL_ABBREVIATIONS.get(model_name, model_name.replace(' ', '_'))
                safe_target_name = TARGET_ABBREVIATIONS.get(TARGETS[target_name], target_name)
                file_path = f"plots/pdp_{safe_model_name}_{safe_target_name}_{safe_feature_1}_{safe_feature_2}.png"
                plt.savefig(file_path, dpi=100, bbox_inches='tight')
                plt.close()
                print(f"Saved PDP plot to {file_path}")
            except KeyError as e:
                print(f"Error: Feature {e} not found in X_columns. Skipping PDP for {model_name}, {target_name}, {feature_1} vs {feature_2}.")
            except Exception as e:
                print(f"Error computing PDP for {model_name}, {target_name}, {feature_1} vs {feature_2}: {e}")

def shap_analysis(X, y, target_name, models, X_train_scaled):
    """Perform SHAP analysis with all features and subsampled data, using Matplotlib"""
    sample_size = min(100, X_train_scaled.shape[0])  
    sample_indices = np.random.choice(X_train_scaled.shape[0], sample_size, replace=False)
    X_train_scaled_sample = X_train_scaled[sample_indices]
    X_train_df_sample = pd.DataFrame(X_train_scaled_sample, columns=X.columns)
    feature_abbrs = get_feature_abbreviations(X.columns)
    print(f"SHAP: Using {sample_size} samples with {X_train_df_sample.shape[1]} features")

    for model_name, model in models.items():
        print(f"Computing SHAP for {model_name} on {target_name}...")
        try:
            print("Initializing SHAP explainer...")
            if model_name == 'Bagging Regressor':
                explainer = shap.KernelExplainer(model.predict, X_train_scaled_sample)
            else:
                explainer = shap.TreeExplainer(model, approximate=True)
            
            print("Computing SHAP values...")
            shap_values = explainer.shap_values(X_train_scaled_sample)

            if shap_values.shape[1] != X_train_df_sample.shape[1]:
                print(f"Error: SHAP values shape {shap_values.shape} does not match X_train_df shape {X_train_df_sample.shape} for {model_name}.")
                continue

            
            print(f"Rendering SHAP summary plot for {model_name} - {target_name}")
            plt.figure(figsize=(8, max(4, len(X.columns) * 0.5)))
            shap.summary_plot(shap_values, X_train_df_sample, feature_names=feature_abbrs, show=False)
            plt.title(f'SHAP Summary: {MODEL_ABBREVIATIONS.get(model_name, model_name)} - {TARGET_ABBREVIATIONS.get(TARGETS[target_name], target_name)}', fontsize=14)
            safe_model_name = MODEL_ABBREVIATIONS.get(model_name, model_name.replace(' ', '_'))
            safe_target_name = TARGET_ABBREVIATIONS.get(TARGETS[target_name], target_name)
            file_path = f'plots/shap_summary_{safe_model_name}_{safe_target_name}.png'
            plt.savefig(file_path, dpi=100, bbox_inches='tight')
            plt.close()
            print(f"Saved SHAP summary plot to {file_path}")

            
            print(f"Rendering SHAP bar plot for {model_name} - {target_name}")
            plt.figure(figsize=(8, max(4, len(X.columns) * 0.5)))
            shap.summary_plot(shap_values, X_train_df_sample, feature_names=feature_abbrs, plot_type="bar", show=False)
            plt.title(f'SHAP Importance: {MODEL_ABBREVIATIONS.get(model_name, model_name)} - {TARGET_ABBREVIATIONS.get(TARGETS[target_name], target_name)}', fontsize=14)
            file_path = f'plots/shap_bar_{safe_model_name}_{safe_target_name}.png'
            plt.savefig(file_path, dpi=100, bbox_inches='tight')
            plt.close()
            print(f"Saved SHAP bar plot to {file_path}")

            print(f"Completed SHAP analysis for {model_name} on {target_name}")
        except Exception as e:
            print(f"Error in SHAP analysis for {model_name} on {target_name}: {e}")

def main():
    """Main function to run SHAP and PDP analysis using pre-trained models"""
    print("Starting SHAP and PDP analysis pipeline...")
    
    os.makedirs('plots', exist_ok=True)
    
    
    file_path = 'ml2.csv'
    data = load_data(file_path)
    if data is None:
        print("Failed to load data. Exiting.")
        return
    
    for target_name, target_col in TARGETS.items():
        print(f"\nProcessing SHAP and PDP for {target_name}...")
        if target_col not in data.columns:
            print(f"Error: Target column {target_col} not found in dataset. Skipping {target_name}.")
            continue
        y = data[target_col].copy()
        y.name = TARGET_ABBREVIATIONS.get(target_col, target_col)
        
        
        models = {}
        for model_name in MODEL_ABBREVIATIONS:
            
            model_file_name = model_name.replace(' ', '_')
            if model_name == 'Gradient Boosting Decision Tree':
                model_file_name = 'Gradient_Boosting_Decision_Tree'
            model_file = f'saved_models/{target_name}_{model_file_name}.pkl'
            try:
                models[model_name] = joblib.load(model_file)
                print(f"Loaded model {model_name} from {model_file}")
                if hasattr(models[model_name], 'feature_names_in_'):
                    print(f"Model expected features: {models[model_name].feature_names_in_}")
            except FileNotFoundError:
                print(f"Error: Model file {model_file} not found. Skipping {model_name} for {target_name}.")
                continue
        
        for model_name in models:
            
            selected_features = load_selected_features(target_name, model_name)
            if selected_features is None:
                print(f"Skipping {model_name} for {target_name} due to missing or invalid feature file")
                continue
            
            
            missing_features = [f for f in selected_features if f not in data.columns]
            if missing_features:
                print(f"Error: Features {missing_features} not found in dataset. Skipping {model_name} for {target_name}.")
                continue
            
            
            data_processed = preprocess_data(data, selected_features)
            if data_processed is None:
                print(f"Failed to preprocess data for {model_name} on {target_name}. Skipping.")
                continue
            
            try:
                X = data_processed[selected_features].copy()
            except KeyError as e:
                print(f"Error: One or more selected features not found in dataset: {e}")
                print(f"Available columns: {list(data.columns)}")
                continue
            
            
            if 'Preparation method' in X.columns:
                print("Encoding categorical feature 'Preparation method'...")
                X = pd.get_dummies(X, columns=['Preparation method'], drop_first=True)
            
            
            try:
                print(f"Splitting data for {model_name} on {target_name}...")
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                print(f"Imputing missing values for {model_name} on {target_name}...")
                imputer = IterativeImputer(random_state=42, max_iter=10)
                X_train_imputed = imputer.fit_transform(X_train)
                X_test_imputed = imputer.transform(X_test)
                
                print(f"Scaling features for {model_name} on {target_name}...")
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train_imputed)
                X_test_scaled = scaler.transform(X_test_imputed)
                print(f"Preprocessed data shape: {X_train_scaled.shape}")
            except Exception as e:
                print(f"Error preprocessing data for {model_name} on {target_name}: {e}")
                continue
            
            
            shap_analysis(X, y, target_name, {model_name: models[model_name]}, X_train_scaled)
            plot_3d_pdp_matplotlib(X_train_scaled, {model_name: models[model_name]}, X.columns, target_name, pdp_feature_pairs)


    print("\nPipeline completed successfully!")

if __name__ == "__main__":
    main()