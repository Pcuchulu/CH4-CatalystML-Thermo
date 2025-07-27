import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.experimental import enable_iterative_imputer
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import partial_dependence
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import os
import plotly.graph_objects as go
import shap
from matplotlib.ticker import FuncFormatter
import warnings
warnings.filterwarnings('ignore')


plt.rcParams['text.usetex'] = False


try:
    import scienceplots
    plt.style.use(['science', 'grid'])
except ImportError:
    print("Warning: 'scienceplots' not installed. Falling back to 'ggplot' style.")
    plt.style.use('ggplot')


pdp_feature_pairs = [
    ('Active component type density', 'Active component type formation energy'),
    ('Promoter type density', 'Promoter type formation energy'),
    ('Support a type density', 'Support a type formation energy'),
    ('Temperature (C)', 'Pressure (bar)')
]


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


file_path = 'SI.csv'  
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


features = [
    'Active component type formation energy',
    'Active component type density',
    'Active component content (wt percent)',
    'Promoter type formation energy',
    'Promoter type density',
    'Promoter content (wt percent)',
    'Support a type formation energy',
    'Support a type density',
    'Support a content (wt percent)',
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
    'Active component type density': 'ACTD',
    'Active component type formation energy': 'ACTFE',
    'Active component content (wt percent)': 'ACW',
    'Promoter type density': 'PTD',
    'Promoter type formation energy': 'PTFE',
    'Promoter content (wt percent)': 'PTW',
    'Support a type density': 'SATD',
    'Support a type formation energy': 'SATFE',
    'Support a content (wt percent)': 'SAW',
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


try:
    X = data[features]
except KeyError as e:
    print(f"Error: One or more feature columns not found in dataset: {e}")
    print(f"Available columns: {list(data.columns)}")
    raise
X.columns = [abbreviations.get(col, col) for col in X.columns]


targets = {
    'co2 conversion ratio (percent)': data['co2 conversion ratio (percent)'],
    'ch4 selectivity (percent)': data['ch4 selectivity (percent)'],
    'ch4 yield (percent)': data['ch4 yield (percent)']
}


target_abbr = {
    'co2 conversion ratio (percent)': 'CO2_CR',
    'ch4 selectivity (percent)': 'CH4_S',
    'ch4 yield (percent)': 'CH4_Y'
}
for target_name, y in targets.items():
    y.name = target_abbr.get(y.name, y.name)


def generate_heatmaps(X, targets, abbreviations):
    for target_name, y in targets.items():
        X_train, _, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        corrmat = X_train.corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            corrmat,
            cmap="rainbow",
            annot=False,
            linewidths=0.5,
            ax=ax,
            xticklabels=X.columns,
            yticklabels=X.columns
        )
        plt.title(f'{target_name.replace(" (percent)", " (%)")} - Feature Correlation')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f"plots/heat_{target_abbr[target_name]}.png", format='png', dpi=600, bbox_inches='tight')
        plt.close()


os.makedirs('saved_models', exist_ok=True)
os.makedirs('plots', exist_ok=True)


generate_heatmaps(X, targets, abbreviations)


def train_evaluate_model(X, y, target_name):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    imputer = IterativeImputer(random_state=42)
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    joblib.dump(imputer, f'saved_models/{target_name}_imputer.pkl')

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)
    joblib.dump(scaler, f'saved_models/{target_name}_scaler.pkl')

    model_results = {}
    models = {}

    def evaluate_model(model, model_name, save_path):
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        model_results[model_name] = {
            'MSE': mean_squared_error(y_test, y_pred),
            'R²': r2_score(y_test, y_pred)
        }
        joblib.dump(model, save_path)
        print(f"{model_name} saved for {target_name} at {save_path}")
        models[model_name] = model

    rf_param_grid = {
        'n_estimators': [100, 300, 500],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    rf_model = RandomizedSearchCV(
        estimator=RandomForestRegressor(random_state=42),
        param_distributions=rf_param_grid,
        n_iter=20, cv=5, verbose=0, n_jobs=-1
    )
    evaluate_model(rf_model, 'Random Forest', f'saved_models/{target_name}_rf.pkl')

    tf_model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        BatchNormalization(),
        Dropout(0.2),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(1)
    ])
    tf_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    tf_model.fit(X_train_scaled, y_train, validation_data=(X_test_scaled, y_test),
                 epochs=400, batch_size=32, verbose=0,
                 callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)])
    y_pred_tf = tf_model.predict(X_test_scaled).flatten()
    model_results['TensorFlow Neural Network'] = {
        'MSE': mean_squared_error(y_test, y_pred_tf),
        'R²': r2_score(y_test, y_pred_tf)
    }
    tf_model.save(f'saved_models/{target_name}_tf_model.keras')
    print(f"TensorFlow model saved for {target_name} at saved_models/{target_name}_tf_model.keras")
    models['TensorFlow Neural Network'] = tf_model

    evaluate_model(Ridge(alpha=1.0), 'Ridge Regression', f'saved_models/{target_name}_ridge.pkl')
    evaluate_model(Lasso(alpha=0.01), 'Lasso Regression', f'saved_models/{target_name}_lasso.pkl')
    evaluate_model(MLPRegressor(hidden_layer_sizes=(128, 64, 32), max_iter=1000, random_state=42),
                  'MLP Neural Network', f'saved_models/{target_name}_mlp.pkl')

    xgb_param_grid = {
        'n_estimators': [100, 300, 500],
        'max_depth': [3, 6, 10],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    xgb_model = RandomizedSearchCV(
        estimator=XGBRegressor(random_state=42, objective='reg:squarederror'),
        param_distributions=xgb_param_grid,
        n_iter=20, cv=5, verbose=0, n_jobs=-1
    )
    evaluate_model(xgb_model, 'XGBoost', f'saved_models/{target_name}_xgb.pkl')

    stack_model = StackingRegressor(
        estimators=[
            ('rf', RandomForestRegressor(n_estimators=300, random_state=42)),
            ('xgb', XGBRegressor(n_estimators=300, random_state=42, objective='reg:squarederror'))
        ],
        final_estimator=LinearRegression()
    )
    evaluate_model(stack_model, 'Stacking Ensemble', f'saved_models/{target_name}_stack.pkl')

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(rf_model.best_estimator_, X_train_scaled, y_train, cv=kfold, scoring='r2')
    model_results['Cross-Validation R²'] = {
        'Mean R²': np.mean(cv_scores),
        'Std Dev': np.std(cv_scores)
    }

    return model_results, X_train_scaled, X_test_scaled, y_train, y_test, models, X_train


def compute_pdp_tf_2d(model, X, feature_idx_1, feature_idx_2, grid_resolution=10):
    X_temp = X.copy()
    feature_values_1 = np.linspace(np.min(X[:, feature_idx_1]), np.max(X[:, feature_idx_1]), grid_resolution)
    feature_values_2 = np.linspace(np.min(X[:, feature_idx_2]), np.max(X[:, feature_idx_2]), grid_resolution)
    pdp_values = np.zeros((grid_resolution, grid_resolution))
    for i, val1 in enumerate(feature_values_1):
        for j, val2 in enumerate(feature_values_2):
            X_temp[:, feature_idx_1] = val1
            X_temp[:, feature_idx_2] = val2
            preds = model.predict(X_temp, verbose=0).flatten()
            pdp_values[i, j] = np.mean(preds)
    return feature_values_1, feature_values_2, pdp_values


def plot_3d_pdp_extended(X_train_scaled, models, X_columns, target_name, feature_pairs):
    for model_name, model in models.items():
        for feature_1, feature_2 in feature_pairs:
            try:
                feature_1_idx = X_columns.get_loc(abbreviations.get(feature_1, feature_1))
                feature_2_idx = X_columns.get_loc(abbreviations.get(feature_2, feature_2))

                
                if model_name == 'TensorFlow Neural Network':
                    grid_values_1, grid_values_2, pdp_values = compute_pdp_tf_2d(model, X_train_scaled, feature_1_idx, feature_2_idx)
                    XX, YY = np.meshgrid(grid_values_1, grid_values_2)
                    Z = pdp_values
                else:
                    pdp_result = partial_dependence(
                        model.best_estimator_ if hasattr(model, 'best_estimator_') else model,
                        X_train_scaled, features=[feature_1_idx, feature_2_idx], kind="average", grid_resolution=10
                    )
                    XX, YY = np.meshgrid(pdp_result["grid_values"][0], pdp_result["grid_values"][1])
                    Z = pdp_result["average"][0].T

                
                fig = plt.figure(figsize=(8, 6))
                ax = fig.add_subplot(111, projection='3d')
                surf = ax.plot_surface(XX, YY, Z, cmap='plasma', edgecolor='none')
                ax.set_xlabel(feature_1.replace(' type', ''), fontsize=8)
                ax.set_ylabel(feature_2.replace(' type', ''), fontsize=8)
                ax.set_zlabel("Partial Dependence", fontsize=8)
                formatter = FuncFormatter(lambda x, _: f'{x:.2f}')
                ax.xaxis.set_major_formatter(formatter)
                ax.yaxis.set_major_formatter(formatter)
                ax.zaxis.set_major_formatter(formatter)
                ax.tick_params(axis='both', which='major', labelsize=8)
                ax.tick_params(axis='z', which='major', labelsize=8)
                fig.colorbar(surf, ax=ax, shrink=0.4, aspect=14, pad=0.12)
                plt.title(f'PDP: {model_name} - {target_name} ({feature_1.replace(" type", "")} vs {feature_2.replace(" type", "")})')
                plt.tight_layout()
                safe_feature_1 = feature_1.replace(' ', '_').replace('(', '').replace(')', '')
                safe_feature_2 = feature_2.replace(' ', '_').replace('(', '').replace(')', '')
                safe_model_name = model_name.replace(' ', '_')
                plt.savefig(f"plots/pdp_{safe_model_name}_{target_abbr[target_name]}_{safe_feature_1}_{safe_feature_2}.png", format='png', dpi=600, bbox_inches='tight')
                plt.close()

            except KeyError as e:
                print(f"Error: Feature {e} not found in X_columns. Skipping PDP for {model_name}, {target_name}, {feature_1} vs {feature_2}.")
            except Exception as e:
                print(f"Error computing PDP for {model_name}, {target_name}, {feature_1} vs {feature_2}: {e}")


def seaborn_shap_analysis(X, y, target_name, models, X_train_scaled, data, X_train):
    X_train_df = pd.DataFrame(X_train_scaled, columns=X.columns)

    
    if target_name == 'ch4 yield (percent)':
        corrmat = X_train.corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            corrmat,
            cmap="rainbow",
            annot=False,
            linewidths=0.5,
            ax=ax,
            xticklabels=X.columns,
            yticklabels=X.columns
        )
        plt.title('CH$_4$ Yield (%) - Feature Correlation')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig("plots/heat.png", format='png', dpi=600, bbox_inches='tight')
        plt.close()

    
    selected_features = ['ACTFE', 'ACTD', 'T', 'P', target_name]
    y_df = pd.DataFrame(y, columns=[target_name])
    sns.pairplot(pd.concat([X, y_df], axis=1)[selected_features], diag_kind='kde')
    plt.savefig(f'plots/pairplot_{target_name}.png', dpi=600, bbox_inches='tight')
    plt.close()

    
    plt.figure(figsize=(8, 6))
    sns.violinplot(x=pd.cut(X['T'], bins=5), y=y, palette='viridis')
    plt.xlabel('Temperature (°C) Bins')
    plt.ylabel(target_name)
    plt.title(f'{target_name} Distribution Across Temperature Bins')
    plt.savefig(f'plots/violin_{target_name}.png', dpi=600, bbox_inches='tight')
    plt.close()

    
    if target_name == 'co2 conversion ratio (percent)':
        data['Unique Count'] = data['Active component type formation energy'].map(
            data['Active component type formation energy'].value_counts())
        
        plt.figure(figsize=(6, 5))
        ax = plt.gca()
        sns.scatterplot(
            data=data,
            x='Temperature (C)',
            y='co2 conversion ratio (percent)',
            hue='Active component type formation energy',
            size='Unique Count',
            sizes=(20, 200),
            palette='rainbow',
            legend=False,
            ax=ax,
            alpha=0.5
        )
        norm = plt.Normalize(vmin=data['Active component type formation energy'].min(),
                            vmax=data['Active component type formation energy'].max())
        sm = plt.cm.ScalarMappable(cmap='rainbow', norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, label='Active Component Formation Energy', shrink=0.8, aspect=10)
        cbar.ax.tick_params(labelsize=10)
        ax.set_xlabel('Temperature (°C)', fontsize=12, weight='bold')
        ax.set_ylabel('CO$_2$ Conversion Ratio (%)', fontsize=12, weight='bold')
        ax.tick_params(axis='both', labelsize=14)
        plt.savefig('plots/scatter_temp_co2.png', format='png', dpi=600, bbox_inches='tight')
        plt.close()

        
        plt.figure(figsize=(6, 5))
        ax = plt.gca()
        sns.scatterplot(
            data=data,
            x='Pressure (bar)',
            y='co2 conversion ratio (percent)',
            hue='Active component type formation energy',
            size='Unique Count',
            sizes=(20, 200),
            palette='rainbow',
            legend=False,
            ax=ax,
            alpha=0.5
        )
        norm = plt.Normalize(vmin=data['Active component type formation energy'].min(),
                            vmax=data['Active component type formation energy'].max())
        sm = plt.cm.ScalarMappable(cmap='rainbow', norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, label='Active Component Formation Energy', shrink=0.8, aspect=10)
        cbar.ax.tick_params(labelsize=10)
        ax.set_xlabel('Pressure (bar)', fontsize=12, weight='bold')
        ax.set_ylabel('CO$_2$ Conversion Ratio (%)', fontsize=12, weight='bold')
        ax.tick_params(axis='both', labelsize=14)
        plt.savefig('plots/scatter_pressure_co2.png', format='png', dpi=600, bbox_inches='tight')
        plt.close()

        
        plot_data = data[['Preparation cost', 'Preparation Scalability', 'co2 conversion ratio (percent)', 'Preparation method']].copy()
        plot_data['Preparation method'] = plot_data['Preparation method'].astype('category')
        print(f"plot_data shape: {plot_data.shape}")
        print(f"Missing values in plot_data:\n{plot_data.isna().sum()}")
        print(f"Sample of plot_data:\n{plot_data.head()}")

        if not plot_data.empty:
            bubble_size = plot_data['Preparation Scalability']
            plt.figure(figsize=(10, 6))
            sns.scatterplot(
                data=plot_data,
                x='Preparation cost',
                y='co2 conversion ratio (percent)',
                size=bubble_size,
                sizes=(50, 500),
                hue='Preparation method',
                palette='viridis',
                alpha=0.8
            )
            plt.xlabel("Preparation Cost", fontsize=14)
            plt.ylabel('CO$_2$ Conversion Ratio (%)', fontsize=14)
            plt.legend(
                loc='upper center',
                bbox_to_anchor=(0.5, 1.3),
                ncol=6,
                frameon=True,
                handlelength=1,
                fontsize=8
            )
            plt.xticks(ticks=plot_data['Preparation cost'].unique(), fontsize=16, fontweight='bold')
            plt.tick_params(axis='y', labelsize=16)
            plt.savefig("plots/3.png", format='png', dpi=600, bbox_inches='tight')
            plt.tight_layout()
            plt.close()
        else:
            print(f"Warning: plot_data is empty for scatter plot (3.png) for {target_name}. Check data loading and column names.")

    
    for model_name, model in models.items():
        print(f"Computing SHAP for {model_name} on {target_name}...")
        X_plot = pd.DataFrame(X_train_scaled, columns=X.columns)
        shap_values = None
        try:
            if model_name == 'Random Forest':
                explainer = shap.TreeExplainer(model.best_estimator_)
                shap_values = explainer.shap_values(X_train_scaled)
            elif model_name == 'XGBoost':
                explainer = shap.TreeExplainer(model.best_estimator_)
                shap_values = explainer.shap_values(X_train_scaled)
            elif model_name in ['Ridge Regression', 'Lasso Regression']:
                explainer = shap.LinearExplainer(model, X_train_scaled)
                shap_values = explainer.shap_values(X_train_scaled)
            elif model_name == 'TensorFlow Neural Network':
                explainer = shap.DeepExplainer(model, X_train_scaled[:100])
                shap_values = explainer.shap_values(X_train_scaled[:100])[0]
                X_plot = X_plot.iloc[:100]
            elif model_name == 'MLP Neural Network':
                explainer = shap.KernelExplainer(model.predict, X_train_scaled[:50])
                shap_values = explainer.shap_values(X_train_scaled[:50], nsamples=100)
                X_plot = X_plot.iloc[:50]
            elif model_name == 'Stacking Ensemble':
                explainer = shap.KernelExplainer(model.predict, X_train_scaled[:50])
                shap_values = explainer.shap_values(X_train_scaled[:50], nsamples=100)
                X_plot = X_plot.iloc[:50]
            else:
                continue

            if shap_values.shape[1] != X_plot.shape[1]:
                print(f"Error: SHAP values shape {shap_values.shape} does not match X_plot shape {X_plot.shape} for {model_name}.")
                continue

            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_plot, feature_names=X.columns, show=False)
            plt.title(f'SHAP Summary Plot for {model_name} - {target_name}')
            plt.savefig(f'plots/shap_summary_{model_name}_{target_name}.png', dpi=600, bbox_inches='tight')
            plt.close()

            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_plot, feature_names=X.columns, plot_type="bar", show=False)
            plt.title(f'SHAP Feature Importance for {model_name} - {target_name}')
            plt.savefig(f'plots/shap_bar_{model_name}_{target_name}.png', dpi=600, bbox_inches='tight')
            plt.close()

            plt.figure(figsize=(12, 8))
            sns.heatmap(shap_values, cmap='viridis', xticklabels=X.columns)
            plt.title(f'SHAP Heatmap for {model_name} - {target_name}')
            plt.xlabel('Features')
            plt.ylabel('Instances')
            plt.xticks(rotation=45, ha='right')
            plt.savefig(f'plots/shap_heatmap_{model_name}_{target_name}.png', dpi=600, bbox_inches='tight')
            plt.close()

        except Exception as e:
            print(f"Error in SHAP analysis for {model_name} on {target_name}: {e}")


def predict_with_mlp(target_name, user_input_dict):
    model_path = f'saved_models/{target_name}_mlp.pkl'
    scaler_path = f'saved_models/{target_name}_scaler.pkl'
    mlp_model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    user_input_df = pd.DataFrame([user_input_dict])
    user_input_df.columns = [abbreviations.get(col, col) for col in user_input_df.columns]
    user_input_scaled = scaler.transform(user_input_df[X.columns].values)
    prediction = mlp_model.predict(user_input_scaled)
    return prediction[0]


results_summary = {}
for target_name, y in targets.items():
    print(f"\nProcessing {target_name}...")
    results, X_train_scaled, X_test_scaled, y_train, y_test, models, X_train = train_evaluate_model(X, y, target_name)
    results_summary[target_name] = results

    seaborn_shap_analysis(X, y, target_name, models, X_train_scaled, data, X_train)
    plot_3d_pdp_extended(X_train_scaled, models, X.columns, target_name, pdp_feature_pairs)


for target_name, results in results_summary.items():
    print(f"\nResults for {target_name}:")
    for model_name, metrics in results.items():
        print(f"{model_name}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        print("-" * 50)


user_input = {
    'Active component type formation energy': 0.046,
    'Active component type density': 9.22,
    'Active component content (wt percent)': 4,
    'Promoter type formation energy': 0,
    'Promoter type density': 0,
    'Promoter content (wt percent)': 0,
    'Support a type formation energy': -3.352,
    'Support a type density': 3.87,
    'Support a content (wt percent)': 90,
    'Calcination Temperature (C)': 500,
    'Calcination time (h)': 3,
    'Reduction Temperature (C)': 500,
    'Reduction Pressure (bar)': 1,
    'Reduction time (h)': 2,
    'Reduced hydrogen content (vol percent)': 100,
    'Temperature (C)': 200,
    'Pressure (bar)': 1,
    'Weight hourly space velocity [mgcat/(min·ml)]': 6.67,
    'Content of inert components in raw materials (vol percent)': 0,
    'h2/co2 ratio (mol/mol)': 4,
    'Preparation Scalability': 4,
    'Preparation cost': 2
}
prediction = predict_with_mlp('co2 conversion ratio (percent)', user_input)
print(f"\nPredicted CO2 Conversion Ratio: {prediction:.2f} %")