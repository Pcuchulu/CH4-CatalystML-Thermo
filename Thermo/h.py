import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from SALib.analyze import sobol
from SALib.sample import saltelli
import os
import re
from matplotlib import cm, gridspec, rcParams
from scipy.interpolate import griddata



plt.rcParams.update({
    "text.usetex": True,  
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 14,  
    "axes.titlesize": 25,  
    "axes.labelsize": 14,  
    "xtick.labelsize": 13,  
    "ytick.labelsize": 13,  
    "legend.fontsize": 12,  
    "figure.titlesize": 16, 
    "mathtext.fontset": "cm",  
    "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}"
})




def safe_filename(name):
    """Convert a string to a safe filename"""
    return re.sub(r'[\\/*?:"<>|]', "_", name)

def load_data():
    try:
        df = pd.read_excel('g.xlsx', sheet_name='sobol')
    except FileNotFoundError:
        df = pd.read_excel('Book1.xlsx', sheet_name='Sheet1')
    
    
    display_names = {
        
        'Reactor Temp': r'Reactor Temp ($^\circ$C)',
        'Pressure': r'Pressure (bar)',
        'Off-Gas Flow': r'Off-Gas Flow (m$^3$/h)',
        'H2 Inlet Flow': r'H$_2$ Inlet Flow (m$^3$/h)',
        
        
        'Total Flow': r'Total Flow (m$^3$/h)',
        'CH4 Flow': r'CH$_4$ Flow (m$^3$/h)',
        'H2O Flow': r'H$_2$O Flow (m$^3$/h)',
        'N2 Flow': r'N$_2$ Flow (m$^3$/h)',
        'CO2 Flow': r'CO$_2$ Flow (m$^3$/h)',
        'CO Flow': r'CO Flow (m$^3$/h)',
        'H2 Flow': r'H$_2$ Flow (m$^3$/h)',
        'Carbon Flow': r'Carbon Flow (m$^3$/h)',
        'CH4 wt': r'CH$_4$ wt.\%',
        'H2O wt': r'H$_2$O wt.\%',
        'N2 wt': r'N$_2$ wt.\%',
        'CO2 wt': r'CO$_2$ wt.\%',
        'CO wt': r'CO wt.\%',
        'H2 wt': r'H$_2$ wt.\%',
        'C wt': r'C wt.\%',
        'Reaction Enthalpy': r'Reaction Enthalpy (kJ/mol)',
        'Total Energy Efficiency': r'Total Energy Efficiency (\%)',
        'CH4 Exergy eficiency': r'CH$_4$ Exergy Efficiency (\%)',
        'CH4 Energy Efficiency': r'CH$_4$ Energy Efficiency (\%)',
        'Total Exergy Efficiency': r'Total Exergy Efficiency (\%)',
        'CO2 Conversion': r'CO$_2$ Conversion (\%)',
        'CO Conversion': r'CO Conversion (\%)',
        'CH4 Selectivity': r'CH$_4$ Selectivity (\%)',
        'CH4 Yield': r'CH$_4$ Yield (\%)',
        'Carbon Yield': r'Carbon Yield (\%)',
        'Carbon Balance': r'Carbon Balance (\%)'
    }
    
    
    inputs = ['Reactor Temp', 'Pressure', 'Off-Gas Flow', 'H2 Inlet Flow']
    outputs = [col for col in df.columns if col not in inputs]
    
    
    missing_cols = [col for col in inputs + outputs if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Columns not found in Excel file: {missing_cols}")
    
    return df, inputs, outputs, display_names

def perform_sobol_analysis(df, inputs, outputs, display_names):
    
    folder_name = "Sobol_Analysis_Results"
    os.makedirs(folder_name, exist_ok=True)
    
    
    problem = {
        'num_vars': len(inputs),
        'names': [display_names[col] for col in inputs],
        'bounds': [[df[col].min(), df[col].max()] for col in inputs]
    }
    
    
    for output in outputs:
        print(f"\nAnalyzing: {output}")
        
        try:
            Y = df[output].values
            param_values = saltelli.sample(problem, 1024, calc_second_order=True)
            
            if len(param_values) > len(Y):
                param_values = param_values[:len(Y)]
            
            Si = sobol.analyze(problem, Y, calc_second_order=True)
            
            
            plot_sobol_results(Si, problem, output, display_names[output], folder_name)
            
            
            for i in range(len(inputs)):
                for j in range(i+1, len(inputs)):
                    create_contour_plot(
                        df, inputs[i], inputs[j], output, 
                        display_names, folder_name
                    )
        except Exception as e:
            print(f"Error analyzing {output}: {str(e)}")
            continue
    
    print(f"\nAnalysis complete. Results saved in: {os.path.abspath(folder_name)}")

def plot_sobol_results(Si, problem, output_col, output_name, folder_name):
    """Create Sobol indices plot with robust file handling"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))  
    
    
    title_fontsize = 14
    label_fontsize = 20
    tick_fontsize = 16
    
    
    y_pos = np.arange(len(problem['names']))
    ax1.barh(y_pos, Si['S1'], xerr=Si['S1_conf'], 
            color='
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(problem['names'], fontsize=tick_fontsize)
    ax1.set_xlabel('First-order Sobol index', fontsize=label_fontsize)
    ax1.set_title('Main effects', pad=10, fontsize=title_fontsize)
    ax1.tick_params(axis='x', labelsize=tick_fontsize)
    
    
    ax2.barh(y_pos, Si['ST'], xerr=Si['ST_conf'],
            color='
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([], fontsize=tick_fontsize)
    ax2.set_xlabel('Total-order Sobol index', fontsize=label_fontsize)
    ax2.set_title('Total effects', pad=10, fontsize=title_fontsize)
    ax2.tick_params(axis='x', labelsize=tick_fontsize)
    
    
    for ax in [ax1, ax2]:
        ax.xaxis.grid(True, linestyle=':', alpha=0.6)
        ax.set_xlim(0, 1.1*max(Si['ST']))
    
    plt.suptitle(f'Sensitivity Analysis for {output_name}', y=1.02, fontsize=16)
    plt.tight_layout()
    
    
    safe_name = safe_filename(output_col)
    output_path = os.path.join(folder_name, f'sobol_{safe_name}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def create_contour_plot(df, x_col, y_col, output_col, display_names, folder_name):
    """Create contour plot with robust path handling"""
    try:
        x = df[x_col].values
        y = df[y_col].values
        z = df[output_col].values
        
        
        xi = np.linspace(x.min(), x.max(), 100)
        yi = np.linspace(y.min(), y.max(), 100)
        zi = griddata((x, y), z, (xi[None,:], yi[:,None]), method='cubic')
        
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        
        title_fontsize = 14
        label_fontsize = 20
        tick_fontsize = 16
        cbar_fontsize = 16
        
        
        contour = ax.contourf(xi, yi, zi, levels=15, cmap='viridis')
        
        
        ax.scatter(x, y, c='white', s=15, alpha=0.7, 
                  edgecolors='black', linewidth=0.5)
        
        
        ax.set_xlabel(display_names[x_col], fontsize=label_fontsize)
        ax.set_ylabel(display_names[y_col], fontsize=label_fontsize)
        
        
        ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
        
        
        cbar = fig.colorbar(contour, ax=ax, pad=0.02)
        cbar.set_label(display_names[output_col], fontsize=label_fontsize)
        cbar.ax.tick_params(labelsize=tick_fontsize)
        
        
        ax.set_xlim(xi.min(), xi.max())
        ax.set_ylim(yi.min(), yi.max())
        
        
        plt.title(
            f'Effect of {display_names[x_col]} and {display_names[y_col]} on {display_names[output_col]}',
            pad=15, fontsize=title_fontsize
        )
        plt.tight_layout()
        
        
        safe_x = safe_filename(x_col)
        safe_y = safe_filename(y_col)
        safe_out = safe_filename(output_col)
        output_path = os.path.join(folder_name, f'contour_{safe_x}_{safe_y}_{safe_out}.png')
        plt.savefig(output_path, dpi=600, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path}")
    except Exception as e:
        print(f"Error creating contour plot for {output_col}: {str(e)}")

if __name__ == "__main__":
    try:
        df, inputs, outputs, display_names = load_data()
        perform_sobol_analysis(df, inputs, outputs, display_names)
    except Exception as e:
        print(f"Error in analysis: {str(e)}")