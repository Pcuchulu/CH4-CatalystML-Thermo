import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
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
    'Bagged Decision Tree Regressor': 'BR',
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
        print(f"Data loaded successfully. Columns: {list(data.columns)}")
        return data
    except UnicodeDecodeError:
        data = pd.read_csv(file_path, encoding='ISO-8859-1')
        print(f"Data loaded successfully with ISO-8859-1 encoding. Columns: {list(data.columns)}")
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

def plot_3d_pdp_plotly(X_train_scaled, models, X_columns, target_name, feature_pairs):
    """Generate interactive 3D Partial Dependence Plots using Plotly"""
    for model_name, model in models.items():
        for feature_1, feature_2 in feature_pairs:
            try:
                feature_1_idx = X_columns.get_loc(feature_1)
                feature_2_idx = X_columns.get_loc(feature_2)
                pdp_result = partial_dependence(
                    model, X_train_scaled, features=[feature_1_idx, feature_2_idx],
                    kind="average", grid_resolution=5
                )
                XX, YY = np.meshgrid(pdp_result["grid_values"][0], pdp_result["grid_values"][1])
                Z = pdp_result["average"][0].T

                fig = go.Figure(data=[
                    go.Surface(
                        x=XX, y=YY, z=Z,
                        colorscale='Plasma',
                        showscale=True,
                        colorbar=dict(title="Partial Dependence")
                    )
                ])
                fig.update_layout(
                    title=f'PDP: {model_name} - {target_name} ({feature_1} vs {feature_2})',
                    scene=dict(
                        xaxis_title=feature_1,
                        yaxis_title=feature_2,
                        zaxis_title="Partial Dependence",
                        xaxis=dict(tickformat=".2f"),
                        yaxis=dict(tickformat=".2f"),
                        zaxis=dict(tickformat=".2f")
                    ),
                    width=800,
                    height=600
                )
                safe_feature_1 = feature_1.replace(' ', '_').replace('(', '').replace(')', '')
                safe_feature_2 = feature_2.replace(' ', '_').replace('(', '').replace(')', '')
                safe_model_name = model_name.replace(' ', '_')
                safe_target_name = TARGET_ABBREVIATIONS.get(TARGETS[target_name], target_name)
                fig.write_html(
                    f"plots/pdp_{safe_model_name}_{safe_target_name}_{safe_feature_1}_{safe_feature_2}.html"
                )
                print(f"Saved PDP plot for {model_name} - {target_name} ({feature_1} vs {feature_2})")
            except KeyError as e:
                print(f"Error: Feature {e} not found in X_columns. Skipping PDP for {model_name}, {target_name}, {feature_1} vs {feature_2}.")
            except Exception as e:
                print(f"Error computing PDP for {model_name}, {target_name}, {feature_1} vs {feature_2}: {e}")

def shap_analysis(X, y, target_name, models, X_train_scaled):
    """Perform SHAP analysis with Plotly for visualization"""
    sample_size = min(50, X_train_scaled.shape[0])  
    sample_indices = np.random.choice(X_train_scaled.shape[0], sample_size, replace=False)
    X_train_scaled_sample = X_train_scaled[sample_indices]
    X_train_df_sample = pd.DataFrame(X_train_scaled_sample, columns=X.columns)

    for model_name, model in models.items():
        print(f"Computing SHAP for {model_name} on {target_name}...")
        try:
            if model_name == 'Bagged Decision Tree Regressor':
                explainer = shap.KernelExplainer(model.predict, X_train_scaled_sample)
            else:
                explainer = shap.TreeExplainer(model, approximate=True)  
            shap_values = explainer.shap_values(X_train_scaled_sample)

            if shap_values.shape[1] != X_train_df_sample.shape[1]:
                print(f"Error: SHAP values shape {shap_values.shape} does not match X_train_df shape {X_train_df_sample.shape} for {model_name}.")
                continue

            
            shap_values_df = pd.DataFrame(shap_values, columns=X.columns)
            shap_mean = np.abs(shap_values).mean(axis=0)
            top_features = X.columns[np.argsort(shap_mean)[::-1]][:10]  
            fig = px.scatter(
                x=shap_values_df[top_features].values.flatten(),
                y=[f for f in top_features for _ in range(sample_size)],
                color=shap_values_df[top_features].values.flatten(),
                color_continuous_scale='RdBu',
                labels={'x': 'SHAP Value', 'y': 'Feature'},
                title=f'SHAP Summary Plot for {model_name} - {target_name}'
            )
            fig.update_layout(width=800, height=600)
            safe_model_name = model_name.replace(' ', '_')
            safe_target_name = TARGET_ABBREVIATIONS.get(TARGETS[target_name], target_name)
            fig.write_html(f'plots/shap_summary_{safe_model_name}_{safe_target_name}.html')
            
            
            feature_importance = pd.DataFrame({
                'Feature': X.columns,
                'Importance': np.abs(shap_values).mean(axis=0)
            }).sort_values('Importance', ascending=False).head(10)
            fig = px.bar(
                feature_importance,
                x='Importance',
                y='Feature',
                orientation='h',
                title=f'SHAP Feature Importance for {model_name} - {target_name}',
                color='Importance',
                color_continuous_scale='Viridis'
            )
            fig.update_layout(width=800, height=600, yaxis={'autorange': 'reversed'})
            fig.write_html(f'plots/shap_bar_{safe_model_name}_{safe_target_name}.html')
            
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
            model_file = f'saved_models/{target_name}_{model_name.replace(" ", "_")}.pkl'
            try:
                models[model_name] = joblib.load(model_file)
                print(f"Loaded model {model_name} from {model_file}")
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
                X = pd.get_dummies(X, columns=['Preparation method'], drop_first=True)
            
            
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                
                imputer = IterativeImputer(random_state=42, max_iter=10)
                X_train_imputed = imputer.fit_transform(X_train)
                X_test_imputed = imputer.transform(X_test)
                
                
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train_imputed)
                X_test_scaled = scaler.transform(X_test_imputed)
            except Exception as e:
                print(f"Error preprocessing data for {model_name} on {target_name}: {e}")
                continue
            
            
            shap_analysis(X, y, target_name, {model_name: models[model_name]}, X_train_scaled)
            plot_3d_pdp_plotly(X_train_scaled, {model_name: models[model_name]}, X.columns, target_name, pdp_feature_pairs)

    print("\nPipeline completed successfully!")

if __name__ == "__main__":
    main()