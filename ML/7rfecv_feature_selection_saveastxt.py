import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_selection import RFECV
from sklearn.ensemble import (RandomForestRegressor, BaggingRegressor, 
                            GradientBoostingRegressor, ExtraTreesRegressor)
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Lasso
from sklearn.metrics import make_scorer, r2_score, mean_squared_error
from sklearn.pipeline import Pipeline
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.base import clone
from matplotlib.ticker import MaxNLocator

class CO2FeatureSelector:
    """
    A comprehensive feature selection framework for CO2 conversion analysis
    with multiple machine learning models and recursive feature elimination.
    """
    
    def __init__(self):
        # Feature name mappings
        self.feature_map = {
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
        
        # Model configurations with proper initialization
        self.models = {
            'RF': RandomForestRegressor(n_estimators=100, random_state=42),
            'BR': BaggingRegressor(
                estimator=DecisionTreeRegressor(random_state=42),
                n_estimators=100,
                random_state=42
            ),
            'GBDT': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'LGBM': LGBMRegressor(n_estimators=100, random_state=42, verbose=-1),
            'XGB': XGBRegressor(n_estimators=100, random_state=42),
            'CB': CatBoostRegressor(iterations=100, random_state=42, verbose=0)
        }
        
        # Visualization settings
        try:
            plt.style.use('seaborn')
        except:
            plt.style.use('default')
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['savefig.bbox'] = 'tight'
        os.makedirs('saved_features', exist_ok=True)
        os.makedirs('plots', exist_ok=True)

    def clean_data(self, value):
        """Convert problematic strings to numeric values"""
        if pd.isna(value):
            return np.nan
        try:
            cleaned = str(value).replace('?', '').replace(',', '.').strip()
            return float(cleaned) if cleaned else np.nan
        except:
            return np.nan

    def load_and_prepare_data(self, file_path):
        """Load and prepare the dataset"""
        try:
            df = pd.read_csv(file_path, encoding='ISO-8859-1')
            
            # Clean all numeric columns
            for col in df.columns:
                df[col] = df[col].apply(self.clean_data)
            
            # Handle target variables
            targets = ['co2 conversion ratio (percent)', 
                      'ch4 selectivity (percent)', 
                      'ch4 yield (percent)']
            
            # Verify targets exist in dataframe
            targets = [t for t in targets if t in df.columns]
            if not targets:
                raise ValueError("No target variables found in the dataset")
            
            # Impute missing target values
            for target in targets:
                imputer = IterativeImputer(random_state=42)
                df[target] = imputer.fit_transform(df[[target]])
            
            # Select available features and rename
            available_features = [col for col in self.feature_map if col in df.columns]
            if not available_features:
                raise ValueError("No features found in the dataset")
                
            X = df[available_features].copy()
            X.columns = [self.feature_map[col] for col in X.columns]
            
            # Create target dictionary
            targets_dict = {
                'CO2 Conversion': df.get('co2 conversion ratio (percent)', None),
                'CH4 Selectivity': df.get('ch4 selectivity (percent)', None),
                'CH4 Yield': df.get('ch4 yield (percent)', None)
            }
            
            # Remove None targets
            targets_dict = {k: v for k, v in targets_dict.items() if v is not None}
            
            return X, targets_dict
        
        except Exception as e:
            raise ValueError(f"Data processing failed: {str(e)}")

    def _save_feature_results(self, target_name, model_name, selected_features):
        """Save selected features to a text file"""
        os.makedirs('saved_features', exist_ok=True)
        filename = f"saved_features/{target_name}_{model_name}_selected_features.txt"
        with open(filename, 'w') as f:
            for feature in selected_features:
                f.write(f"{feature}\n")

    def run_feature_selection(self, X, y, model, model_name, target_name):
        """Perform recursive feature elimination with cross-validation"""
        try:
            # Ensure data alignment
            X = X.reset_index(drop=True)
            y = y.reset_index(drop=True)
            
            # Check for NaN values in targets
            if y.isna().any():
                y = y.fillna(y.mean())
            
            # Create preprocessing pipeline
            preprocessor = Pipeline([
                ('imputer', IterativeImputer(random_state=42)),
                ('scaler', StandardScaler())
            ])
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Preprocess data
            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)
            
            # Special handling for BaggingRegressor
            if model_name == 'BR':
                # Create a custom importance getter for BaggingRegressor
                def br_importance_getter(estimator):
                    # Average feature importances across all base estimators
                    importances = np.mean([
                        tree.feature_importances_ 
                        for tree in estimator.estimators_
                    ], axis=0)
                    return importances
                
                rfecv = RFECV(
                    estimator=model,
                    step=1,
                    cv=KFold(5, shuffle=True, random_state=42),
                    scoring='r2',
                    min_features_to_select=5,
                    importance_getter=br_importance_getter
                )
            else:
                rfecv = RFECV(
                    estimator=model,
                    step=1,
                    cv=KFold(5, shuffle=True, random_state=42),
                    scoring='r2',
                    min_features_to_select=5,
                    n_jobs=-1
                )
            
            # Fit RFECV
            rfecv.fit(X_train_transformed, y_train)
            
            # Cross-validation metrics
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            r2_scores = []
            rmse_scores = []
            
            for n_features in range(1, X_train_transformed.shape[1] + 1):
                if n_features >= 5:  # Respect minimum features
                    selected = rfecv.support_ if n_features == rfecv.n_features_ else \
                        np.argsort(rfecv.ranking_)[:n_features]
                    
                    X_subset = X_train_transformed[:, selected]
                    
                    fold_r2 = []
                    fold_rmse = []
                    for train_idx, val_idx in kf.split(X_subset):
                        X_train_fold, X_val_fold = X_subset[train_idx], X_subset[val_idx]
                        y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
                        
                        model_clone = clone(model)
                        model_clone.fit(X_train_fold, y_train_fold)
                        y_pred = model_clone.predict(X_val_fold)
                        
                        fold_r2.append(r2_score(y_val_fold, y_pred))
                        fold_rmse.append(np.sqrt(mean_squared_error(y_val_fold, y_pred)))
                    
                    r2_scores.append(np.mean(fold_r2))
                    rmse_scores.append(np.mean(fold_rmse))
                else:
                    r2_scores.append(np.nan)
                    rmse_scores.append(np.nan)
            
            # Save preprocessing pipeline
            joblib.dump(
                preprocessor, 
                f'saved_features/{target_name}_{model_name}_preprocessor.pkl'
            )
            
            # Save selected features
            selected_features = X.columns[rfecv.support_].tolist()
            self._save_feature_results(target_name, model_name, selected_features)
            
            # Visualize results
            self._visualize_results(
                r2_scores, rmse_scores, rfecv, model_name, target_name
            )
            
            return {
                'selected_features': selected_features,
                'n_features': rfecv.n_features_,
                'best_r2': max(r2_scores),
                'best_rmse': min(rmse_scores),
                'ranking': rfecv.ranking_,
                'r2_scores': r2_scores,
                'rmse_scores': rmse_scores
            }
            
        except Exception as e:
            print(f"Error in {model_name} for {target_name}: {str(e)}")
            return None

    def _visualize_results(self, r2_scores, rmse_scores, rfecv, model_name, target_name):
        """Visualize feature selection results with proper integer x-axis ticks"""
        fig, ax1 = plt.subplots(figsize=(8, 5))
        
        n_features = range(1, len(r2_scores) + 1)
        
        # R² plot
        ax1.plot(n_features, r2_scores, 'o-', color='#1f77b4', 
                linewidth=2, markersize=6, label='R²')
        ax1.set_xlabel('Number of Features', fontsize=12)
        ax1.set_ylabel('R² Score', fontsize=12, color='#1f77b4')
        ax1.tick_params(axis='y', labelcolor='#1f77b4')
        
        # RMSE plot
        ax2 = ax1.twinx()
        ax2.plot(n_features, rmse_scores, 's-', color='#ff7f0e', 
                 linewidth=2, markersize=6, label='RMSE')
        ax2.set_ylabel('RMSE', fontsize=12, color='#ff7f0e')
        ax2.tick_params(axis='y', labelcolor='#ff7f0e')
        
        # Set integer ticks on x-axis
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        # Optimal feature count
        ax1.axvline(x=rfecv.n_features_, color='gray', linestyle='--', alpha=0.7)
        
        # Legend and title
        fig.suptitle(f'{model_name} - {target_name}', y=1.02)
        fig.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), 
                  ncol=2, fontsize=10)
        
        plt.tight_layout()
        plt.savefig(f'plots/{target_name}_{model_name}_feature_selection.png', dpi=300)
        plt.close()

    def analyze_all_targets(self, file_path):
        """Run analysis for all targets and models"""
        print("Loading and preparing data...")
        try:
            X, targets = self.load_and_prepare_data(file_path)
        except Exception as e:
            print(f"Failed to load data: {str(e)}")
            return None
        
        results = {}
        for target_name, y in targets.items():
            results[target_name] = {}
            print(f"\nAnalyzing {target_name}...")
            
            for model_name, model in self.models.items():
                print(f"  Running {model_name}...")
                results[target_name][model_name] = self.run_feature_selection(
                    X, y, model, model_name, target_name
                )
        
        self._summarize_results(results)
        return results

    def _summarize_results(self, results):
        """Print and save summary of results"""
        print("\n=== ANALYSIS SUMMARY ===")
        
        with open('feature_selection_summary.txt', 'w') as f:
            for target, models in results.items():
                print(f"\n--- {target} ---")
                f.write(f"\n--- {target} ---\n")
                
                for model, data in models.items():
                    if data:
                        print(f"{model}:")
                        print(f"  Optimal features: {data['n_features']}")
                        print(f"  Best R²: {data['best_r2']:.4f}")
                        print(f"  Best RMSE: {data['best_rmse']:.4f}")
                        print(f"  Selected: {', '.join(data['selected_features'])}")
                        
                        f.write(f"{model}:\n")
                        f.write(f"  Optimal features: {data['n_features']}\n")
                        f.write(f"  Best R²: {data['best_r2']:.4f}\n")
                        f.write(f"  Best RMSE: {data['best_rmse']:.4f}\n")
                        f.write(f"  Selected: {', '.join(data['selected_features'])}\n\n")
                    else:
                        print(f"{model}: Failed")
                        f.write(f"{model}: Failed\n")

def main():
    try:
        analyzer = CO2FeatureSelector()
        results = analyzer.analyze_all_targets('ml2.csv')
        return results
    except Exception as e:
        print(f"An error occurred during analysis: {str(e)}")
        return None

if __name__ == "__main__":
    main()