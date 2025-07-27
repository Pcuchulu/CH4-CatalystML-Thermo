import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
import optuna
import matplotlib.pyplot as plt
import scipy.stats as stats
import pickle
import re


CONFIG = {
    'data_path': 'ml2.csv',
    'feature_selection_dir': 'saved_features',
    'model_dir': 'saved_models',
    'plot_dir': 'plots',
    'test_size': 0.2,
    'random_state': 42,
    'n_trials': 50
}


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
    'Gradient Boosting Decision Tree': 'GBDT'
}


TARGETS = {
    'CO2_Conversion': 'co2 conversion ratio (percent)',
    'CH4_Selectivity': 'ch4 selectivity (percent)',
    'CH4_Yield': 'ch4 yield (percent)'
}

def setup_directories():
    """Create necessary directories if they don't exist"""
    print("Setting up directories...")
    for dir_path in [CONFIG['feature_selection_dir'], CONFIG['model_dir'], CONFIG['plot_dir']]:
        try:
            os.makedirs(dir_path, exist_ok=True)
            if os.path.exists(dir_path):
                print(f"Created and verified directory: {dir_path}")
            else:
                print(f"Failed to create directory: {dir_path}")
        except Exception as e:
            print(f"Error creating directory {dir_path}: {e}")

def load_data():
    """Load dataset with appropriate encoding"""
    print("Attempting to load data...")
    data_path = CONFIG['data_path']
    encodings = ['utf-8', 'latin1']
    
    for encoding in encodings:
        try:
            data = pd.read_csv(data_path, encoding=encoding)
            print(f"Data loaded successfully with {encoding} encoding")
            print(f"Dataset columns: {list(data.columns)}")
            return data
        except UnicodeDecodeError:
            print(f"Trying encoding: {encoding}")
            continue
        except Exception as e:
            print(f"Error loading data with {encoding}: {e}")
            continue
    
    raise Exception("Failed to load data with any encoding")

def load_features(target_name, model_name):
    """Load selected features for target and model from saved_features directory"""
    model_abbr = MODEL_ABBREVIATIONS.get(model_name, model_name.replace(' ', '_'))
    feature_file = os.path.join(CONFIG['feature_selection_dir'], f"{target_name}_{model_abbr}_selected_features.txt")
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

def preprocess_data(data):
    """Preprocess the dataset"""
    print("Preprocessing data...")
    try:
        data = data.copy()
        
        print("Imputing missing target values...")
        for target_col in TARGETS.values():
            if target_col in data.columns:
                data[target_col] = data[target_col].fillna(data[target_col].mean())
        return data
    except Exception as e:
        print(f"Error preprocessing data: {e}")
        return None

def clean_numeric_column(series):
    """Convert series to numeric, handling strings like '?8.83', without removing data"""
    def clean_value(val):
        if isinstance(val, str):
            
            cleaned = re.sub(r'[^\d.-]', '', val)
            try:
                return float(cleaned)
            except ValueError:
                return np.nan  
        return val
    
    cleaned_series = series.apply(clean_value)
    
    if cleaned_series.isna().any():
        mean_value = cleaned_series.mean()
        cleaned_series = cleaned_series.fillna(mean_value)
        print(f"Imputed NaN values with column mean: {mean_value}")
    return cleaned_series

def get_model_config():
    """Return model configuration for Gradient Boosting Decision Tree only"""
    return [
        {
            'name': 'Gradient Boosting Decision Tree',
            'constructor': GradientBoostingRegressor,
            'params': {
                'n_estimators': (50, 500),
                'learning_rate': (0.001, 0.2, 'log'),
                'max_depth': (3, 10),
                'min_samples_split': (2, 20)
            }
        }
    ]

def train_model(X_train, y_train, config):
    """Train a model with Optuna hyperparameter optimization"""
    def objective(trial):
        params = {}
        for param, value in config['params'].items():
            if isinstance(value, tuple):
                if len(value) == 3 and value[2] == 'log':
                    params[param] = trial.suggest_float(param, value[0], value[1], log=True)
                else:
                    params[param] = trial.suggest_int(param, value[0], value[1])
            else:
                params[param] = value
        model = config['constructor'](**params, random_state=CONFIG['random_state'])
        model.fit(X_train, y_train)
        y_pred = model.predict(X_train)
        return r2_score(y_train, y_pred)
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=CONFIG['n_trials'])
    best_params = study.best_params
    model = config['constructor'](**best_params, random_state=CONFIG['random_state'])
    model.fit(X_train, y_train)
    return model, best_params, study

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and return metrics"""
    try:
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        return {'r2': r2, 'rmse': rmse, 'predictions': y_pred}
    except Exception as e:
        print(f"Error evaluating model: {e}")
        return None

def save_model(model, target_name, model_name):
    """Save the trained model"""
    model_filename = os.path.join(CONFIG['model_dir'], f"{target_name}_{model_name.replace(' ', '_')}.pkl")
    try:
        with open(model_filename, 'wb') as f:
            pickle.dump(model, f)
        print(f"Saving model to {model_filename}...")
        print("Model saved successfully")
    except Exception as e:
        print(f"Error saving model to {model_filename}: {e}")

def plot_results(study, y_test, y_pred, model_name, target_name, n_features):
    """Generate and save Q-Q and Predicted vs Actual plots with grid"""
    print(f"Generating plots for {model_name}...")
    try:
        plot_dir = CONFIG['plot_dir']
        if not os.path.exists(plot_dir) or not os.access(plot_dir, os.W_OK):
            print(f"Error: Plot directory {plot_dir} is not writable")
            return
        
        
        plt.figure(figsize=(8, 6))
        stats.probplot(y_test - y_pred, dist="norm", plot=plt)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.title(f'Q-Q Plot for {model_name} ({target_name}, {n_features} features)')
        qq_filename = os.path.join(plot_dir, f"{target_name}_{model_name.replace(' ', '_')}_qq.png")
        plt.savefig(qq_filename)
        plt.close()
        print(f"Saved Q-Q plot to {qq_filename}")
        
        
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'Predicted vs Actual for {model_name} ({target_name}, {n_features} features)')
        pred_vs_actual_filename = os.path.join(plot_dir, f"{target_name}_{model_name.replace(' ', '_')}_pred_vs_actual.png")
        plt.savefig(pred_vs_actual_filename)
        plt.close()
        print(f"Saved Predicted vs Actual plot to {pred_vs_actual_filename}")
        
        print("Plot generation attempt completed")
    except Exception as e:
        print(f"Error generating plots: {e}")

def process_target(data, target_name, target_col):
    """Complete processing for one target variable"""
    print(f"\n{'='*50}")
    print(f"Processing target: {target_name}")
    print(f"{'='*50}")
    
    model_configs = get_model_config()
    target_results = {}
    optuna_best_values = {}
    
    for config in model_configs:
        model_name = config['name']
        print(f"\n=== Training {model_name} ===")
        
        features = load_features(target_name, model_name)
        if features is None:
            print(f"Skipping {model_name} for {target_name} due to missing or invalid feature file")
            optuna_best_values[model_name] = None
            continue
        
        print(f"Using {len(features)} features for {model_name}")
        try:
            X = data[features].copy()  
            y = data[target_col]
        except KeyError as e:
            print(f"Error: Feature or target column not found: {e}")
            print(f"Dataset columns: {list(data.columns)}")
            optuna_best_values[model_name] = None
            continue
        
        
        print(f"Cleaning feature columns for {model_name}...")
        try:
            for col in X.columns:
                if X[col].dtype == 'object':
                    print(f"Non-numeric values detected in column {col}: {X[col].unique()[:5]}")
                    X[col] = clean_numeric_column(X[col])
                
                X[col] = pd.to_numeric(X[col], errors='coerce')
                if X[col].isna().any():
                    mean_value = X[col].mean()
                    X[col] = X[col].fillna(mean_value)
                    print(f"Imputed NaN in column {col} with mean: {mean_value}")
            X = X.astype(float)  
        except Exception as e:
            print(f"Error cleaning features for {model_name}: {e}")
            optuna_best_values[model_name] = None
            continue
        
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=CONFIG['test_size'],
                random_state=CONFIG['random_state']
            )
        except Exception as e:
            print(f"Error splitting data for {model_name}: {e}")
            optuna_best_values[model_name] = None
            continue
        
        print(f"Scaling features for {model_name}...")
        try:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
        except Exception as e:
            print(f"Error scaling features for {model_name}: {e}")
            optuna_best_values[model_name] = None
            continue
        
        try:
            model, best_params, study = train_model(X_train_scaled, y_train, config)
            
            optuna_best_values[model_name] = study.best_value if study.trials else None
            
            eval_results = evaluate_model(model, X_test_scaled, y_test)
            if eval_results is None:
                print(f"Skipping {model_name} due to evaluation error")
                optuna_best_values[model_name] = None
                continue
            
            target_results[model_name] = {
                'params': best_params,
                'r2': eval_results['r2'],
                'rmse': eval_results['rmse']
            }
            
            save_model(model, target_name, model_name)
            plot_results(
                study, y_test, eval_results['predictions'],
                model_name, target_name, len(features)
            )
            
            print(f"Completed {model_name} with R²: {eval_results['r2']:.4f}")
        except Exception as e:
            print(f"Error training {model_name}: {e}")
            optuna_best_values[model_name] = None
            continue
    
    return target_results, optuna_best_values

def main():
    """Main function to run the ML pipeline"""
    print("Starting machine learning pipeline...")
    
    setup_directories()
    
    data = load_data()
    if data is None:
        print("Failed to load data. Exiting.")
        return
    
    data = preprocess_data(data)
    if data is None:
        print("Failed to preprocess data. Exiting.")
        return
    
    all_results = {}
    all_optuna_values = {}
    
    for target_name, target_col in TARGETS.items():
        if target_col not in data.columns:
            print(f"Target column {target_col} not found in data. Skipping {target_name}")
            continue
        results, optuna_values = process_target(data, target_name, target_col)
        all_results[target_name] = results
        all_optuna_values[target_name] = optuna_values
    
    print("\nPipeline completed successfully!")
    
    print("\nFinal Results Summary:")
    for target_name in TARGETS:
        print(f"\n{target_name}:")
        results = all_results.get(target_name, {})
        if results:
            print(f"{'Model':<30} {'R²':<10} {'RMSE':<10}")
            for model_name, metrics in results.items():
                print(f"{model_name:<30} {metrics['r2']:<10.4f} {metrics['rmse']:<10.4f}")
        else:
            print("No results available")
    
    print("\nOptuna Best Optimization Values (R² from Cross-Validation):")
    for target_name in TARGETS:
        print(f"\n{target_name}:")
        optuna_values = all_optuna_values.get(target_name, {})
        for model_name, value in optuna_values.items():
            if value is not None:
                print(f"  {model_name}: {value:.4f}")
            else:
                print(f"  {model_name}: Optimization failed or no trials completed")

if __name__ == "__main__":
    main()