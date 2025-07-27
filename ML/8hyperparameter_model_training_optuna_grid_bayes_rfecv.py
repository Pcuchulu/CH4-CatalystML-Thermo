import pandas as pd
import numpy as np
import optuna
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import joblib
import os
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import scipy.stats as stats
import warnings

warnings.filterwarnings('ignore')


CONFIG = {
    'data_path': 'ml2.csv',
    'feature_selection_dir': 'saved_features',
    'model_save_dir': 'saved_models',
    'plot_save_dir': 'plots',
    'n_trials': 500,
    'test_size': 0.2,
    'random_state': 42,
    'cv_folds': 5
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

TARGETS = {
    'CO2_Conversion': 'co2 conversion ratio (percent)',
    'CH4_Selectivity': 'ch4 selectivity (percent)',
    'CH4_Yield': 'ch4 yield (percent)'
}


MODEL_ABBREVIATIONS = {
    'Random Forest': 'RF',
    'Bagging Regressor': 'BR',
    'Gradient Boosting': 'GBDT',
    'LightGBM': 'LGBM',
    'XGBoost': 'XGB',
    'CatBoost': 'CB'
}

def setup_directories():
    """Create required directories and verify write access"""
    print("\nSetting up directories...")
    for directory in [CONFIG['feature_selection_dir'], CONFIG['model_save_dir'], CONFIG['plot_save_dir']]:
        try:
            os.makedirs(directory, exist_ok=True)
            
            test_file = os.path.join(directory, "test_write.txt")
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
            print(f"Created and verified directory: {directory}")
        except Exception as e:
            print(f"Error creating/verifying directory {directory}: {e}")
            raise

def load_data():
    """Load data with encoding fallback"""
    print("\nAttempting to load data...")
    encodings = ['utf-8', 'latin1', 'ISO-8859-1', 'cp1252']
    for encoding in encodings:
        try:
            print(f"Trying encoding: {encoding}")
            data = pd.read_csv(CONFIG['data_path'], encoding=encoding)
            print("Data loaded successfully")
            return data
        except UnicodeDecodeError:
            continue
    try:
        data = pd.read_csv(CONFIG['data_path'])
        print("Data loaded successfully (no encoding specified)")
        return data
    except Exception as e:
        print(f"Failed to load data: {e}")
        raise

def preprocess_data(data):
    """Clean and preprocess data"""
    print("\nPreprocessing data...")
    
    
    data = data.replace(['?', 'NA', 'nan', 'NaN'], np.nan)
    
    
    numeric_cols = [col for col in data.columns if col != 'Preparation method']
    try:
        data[numeric_cols] = data[numeric_cols].apply(pd.to_numeric, errors='coerce')
    except Exception as e:
        print(f"Error converting numeric columns: {e}")
    
    
    if 'Preparation method' in data.columns:
        try:
            data['Preparation method'] = data['Preparation method'].astype('category')
        except Exception as e:
            print(f"Error handling categorical column 'Preparation method': {e}")
    
    
    print("Imputing missing target values...")
    imputer = IterativeImputer(random_state=CONFIG['random_state'])
    for target in TARGETS.values():
        if target in data.columns:
            try:
                data[target] = imputer.fit_transform(data[[target]])
            except Exception as e:
                print(f"Error imputing target {target}: {e}")
    
    return data

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
        
def get_model_config():
    """Return model configurations with hyperparameter spaces"""
    return [
        {
            'name': 'Random Forest',
            'constructor': RandomForestRegressor,
            'params': {
                'n_estimators': (100, 1000),
                'max_depth': (3, 50),
                'min_samples_split': (2, 20),
                'min_samples_leaf': (1, 10)
            }
        },
        {
            'name': 'Bagging Regressor',
            'constructor': BaggingRegressor,
            'params': {
                'n_estimators': (50, 500),
                'max_samples': (0.1, 1.0),
                'max_features': (0.1, 1.0)
            }
        },
        {
            'name': 'Gradient Boosting',
            'constructor': GradientBoostingRegressor,
            'params': {
                'n_estimators': (50, 500),
                'learning_rate': (0.001, 0.2, 'log'),
                'max_depth': (3, 10),
                'min_samples_split': (2, 20)
            }
        },
        {
            'name': 'LightGBM',
            'constructor': LGBMRegressor,
            'params': {
                'n_estimators': (50, 500),
                'learning_rate': (0.001, 0.2, 'log'),
                'num_leaves': (20, 150),
                'max_depth': (3, 12)
            }
        },
        {
            'name': 'XGBoost',
            'constructor': XGBRegressor,
            'params': {
                'n_estimators': (50, 500),
                'learning_rate': (0.001, 0.2, 'log'),
                'max_depth': (3, 12),
                'subsample': (0.5, 1.0)
            }
        },
        {
            'name': 'CatBoost',
            'constructor': CatBoostRegressor,
            'params': {
                'iterations': (50, 500),
                'learning_rate': (0.001, 0.2, 'log'),
                'depth': (3, 10),
                'l2_leaf_reg': (1, 10)
            }
        }
    ]

def train_model(X_train, y_train, model_config):
    """Train a single model with Optuna optimization"""
    print(f"\nStarting optimization for {model_config['name']}...")
    
    def objective(trial):
        params = {}
        for param, space in model_config['params'].items():
            if isinstance(space, tuple):
                if len(space) == 3 and space[2] == 'log':
                    params[param] = trial.suggest_float(param, space[0], space[1], log=True)
                elif isinstance(space[0], float):
                    params[param] = trial.suggest_float(param, *space[:2])
                else:
                    params[param] = trial.suggest_int(param, *space[:2])
        
        model = model_config['constructor'](**params, random_state=CONFIG['random_state'])
        try:
            return cross_val_score(
                model, X_train, y_train, 
                cv=CONFIG['cv_folds'], 
                scoring='r2',
                n_jobs=-1
            ).mean()
        except Exception as e:
            print(f"Error in cross-validation for {model_config['name']}: {e}")
            return -float('inf')
    
    try:
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(),
            pruner=optuna.pruners.HyperbandPruner()
        )
        
        print(f"Running {CONFIG['n_trials']} trials for {model_config['name']}...")
        study.optimize(objective, n_trials=CONFIG['n_trials'])
        if not study.trials:
            raise ValueError("No trials completed in Optuna study")
        print(f"Completed optimization for {model_config['name']}")
        
        
        print(f"Training final {model_config['name']} model...")
        best_model = model_config['constructor'](
            **study.best_params,
            random_state=CONFIG['random_state']
        )
        best_model.fit(X_train, y_train)
        
        return best_model, study.best_params, study
    except Exception as e:
        print(f"Error training {model_config['name']}: {e}")
        raise

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    print("Evaluating model...")
    try:
        y_pred = model.predict(X_test)
        return {
            'r2': r2_score(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'predictions': y_pred
        }
    except Exception as e:
        print(f"Error evaluating model: {e}")
        return None

def save_model(model, target_name, model_name):
    """Save trained model to disk in saved_models directory"""
    filename = os.path.join(CONFIG['model_save_dir'], f"{target_name.replace(' ', '_')}_{model_name.replace(' ', '_')}.pkl")
    print(f"Saving model to {filename}...")
    try:
        joblib.dump(model, filename)
        print("Model saved successfully")
    except Exception as e:
        print(f"Error saving model to {filename}: {e}")

def plot_results(study, y_test, y_pred, model_name, target_name, n_features):
    """Generate and save Matplotlib plots (Q-Q and Predicted vs Actual)"""
    print(f"Generating plots for {model_name}...")

    
    plot_dir = CONFIG['plot_save_dir']
    try:
        os.makedirs(plot_dir, exist_ok=True)
        test_file = os.path.join(plot_dir, "test_write.txt")
        with open(test_file, 'w') as f:
            f.write("test")
        os.remove(test_file)
        print(f"Plot directory {plot_dir} is writable")
    except Exception as e:
        print(f"Error: Cannot write to plot directory {plot_dir}: {e}")
        return

    
    def save_matplotlib_plot(fig, filename, plot_type):
        full_path = os.path.join(plot_dir, filename)
        try:
            fig.savefig(full_path, dpi=300, bbox_inches='tight', format='png')
            plt.close(fig)
            print(f"Saved {plot_type} plot to {full_path}")
        except Exception as e:
            print(f"Error saving {plot_type} plot to {filename}: {e}")
            plt.close(fig)

    
    print("Generating Q-Q plot...")
    try:
        residuals = y_test - y_pred
        fig, ax = plt.subplots(figsize=(8, 6))
        stats.probplot(residuals, dist="norm", plot=ax)
        ax.set_title(f'Q-Q Plot: {model_name} ({target_name})\nFeatures: {n_features}')
        ax.grid(True, linestyle='--', alpha=0.7)
        save_matplotlib_plot(fig, f"{target_name.replace(' ', '_')}_{model_name.replace(' ', '_')}_qq.png", "Q-Q")
    except Exception as e:
        print(f"Error generating Q-Q plot: {e}")
        plt.close()

    
    print("Generating Predicted vs Actual plot...")
    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(y_test, y_pred, alpha=0.6)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r')
        ax.set_title(f'Predicted vs Actual: {model_name} ({target_name})')
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.grid(True, linestyle='--', alpha=0.7)
        save_matplotlib_plot(fig, f"{target_name.replace(' ', '_')}_{model_name.replace(' ', '_')}_pred_vs_actual.png", "Predicted vs Actual")
    except Exception as e:
        print(f"Error generating Predicted vs Actual plot: {e}")
        plt.close()

    print("Plot generation attempt completed")

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
            X = data[features]
            y = data[target_col]
        except KeyError as e:
            print(f"Error: Feature or target column not found: {e}")
            print(f"Dataset columns: {list(data.columns)}")
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
    """Main execution function"""
    print("Starting machine learning pipeline...")
    
    try:
        
        setup_directories()
        
        
        data = load_data()
        data = preprocess_data(data)
        
        results = {}
        all_optuna_best_values = {}
        
        
        for target_name, target_col in TARGETS.items():
            if target_col not in data.columns:
                print(f"\nTarget column '{target_col}' not found, skipping {target_name}")
                continue
                
            target_results, optuna_best_values = process_target(data, target_name, target_col)
            if target_results is None:
                print(f"Skipping {target_name} due to previous errors")
                continue
            results[target_name] = target_results
            all_optuna_best_values[target_name] = optuna_best_values
            
            
            print(f"\nSummary for {target_name}:")
            print("{:<20} {:<10} {:<15}".format("Model", "R²", "RMSE"))
            for model_name, metrics in target_results.items():
                print("{:<20} {:<10.4f} {:<15.4f}".format(
                    model_name, metrics['r2'], metrics['rmse']))
        
        
        print("\nPipeline completed successfully!")
        print("\nFinal Results Summary:")
        for target, models in results.items():
            print(f"\n{target}:")
            for model, metrics in models.items():
                print(f"  {model}: R²={metrics['r2']:.4f}, RMSE={metrics['rmse']:.4f}")
        
        print("\nOptuna Best Optimization Values (R² from Cross-Validation):")
        for target, models in all_optuna_best_values.items():
            print(f"\n{target}:")
            for model, best_value in models.items():
                if best_value is not None:
                    print(f"  {model}: Best R²={best_value:.4f}")
                else:
                    print(f"  {model}: Optimization failed or no trials completed")
                
    except Exception as e:
        print(f"\nFatal error in pipeline: {e}")
        raise

if __name__ == "__main__":
    main()