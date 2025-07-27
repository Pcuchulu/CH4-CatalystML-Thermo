import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl


mpl.rcParams['font.family'] = 'Times New Roman'  
mpl.rcParams['font.size'] = 12  
mpl.rcParams['axes.linewidth'] = 1.5  
mpl.rcParams['xtick.direction'] = 'in'  
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.major.size'] = 6
mpl.rcParams['ytick.major.size'] = 6
mpl.rcParams['xtick.major.width'] = 1.5
mpl.rcParams['ytick.major.width'] = 1.5


tick_font = 'Times New Roman'  
tick_fontsize = 14  
tick_fontweight = 'bold'  


show_grid = False  


fig_width = 8  
fig_height = 5  


legends = [
    r'CO$_2$(g) + 4H$_2$(g) $\rightarrow$ CH$_4$(g) + 2H$_2$O(g)',
    r'CO(g) + 3H$_2$(g) $\rightarrow$ CH$_4$(g) + H$_2$O(g)',
    r'CO(g) + H$_2$O(g) $\rightarrow$ CO$_2$(g) + H$_2$(g)',
    r'2CO(g) $\rightarrow$ CO$_2$(g) + C(s)',
    r'CO(g) + H$_2$(g) $\rightarrow$ H$_2$O(g) + C(s)',
    r'CO$_2$(g) + 2H$_2$(g) $\rightarrow$ 2H$_2$O(g) + C(s)'
]


markers = ['o', 's', '^', 'D', 'v', 'p']


legend_fontsize = 8  
legend_loc = 'center left'  
legend_bbox = (1, 0.85)  
legend_ncol = 1 
legend_frameon = True  
legend_edgecolor = 'black'  


file_path = 'Thermodynamics data.xlsx'
sheet_name = 'DG,LOGK,DH'


def create_plot(columns, y_label, filename, title, legend_outside=False):
    data = pd.read_excel(file_path, sheet_name=sheet_name, usecols=columns, skiprows=6, nrows=20)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=100)
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize, 
                   labelfontfamily=tick_font, labelcolor='black',
                   width=mpl.rcParams['xtick.major.width'], 
                   length=mpl.rcParams['xtick.major.size'],
                   direction='in')
    
    
    for label in ax.get_xticklabels():
        label.set_fontfamily(tick_font)
        label.set_fontsize(tick_fontsize)
        label.set_fontweight(tick_fontweight)
    for label in ax.get_yticklabels():
        label.set_fontfamily(tick_font)
        label.set_fontsize(tick_fontsize)
        label.set_fontweight(tick_fontweight)
    
    for i, col in enumerate(columns[1:], 1):
        ax.plot(data.iloc[:, 0], data.iloc[:, i], marker=markers[i-1], 
                linestyle='-', linewidth=1.8, markersize=7, label=legends[i-1])
    
    ax.set_xlabel('Temperature (°C)', fontsize=16, fontfamily='Times New Roman', fontweight = 'bold' )
    ax.set_ylabel(y_label, fontsize=16, fontfamily='Times New Roman', fontweight = 'bold')
    if legend_outside:
        ax.legend(fontsize=legend_fontsize, loc=legend_loc, bbox_to_anchor=legend_bbox, 
                  ncol=legend_ncol, frameon=legend_frameon, edgecolor=legend_edgecolor)
    else:
        ax.legend(fontsize=10, loc='best', frameon=True, edgecolor='black')
    if show_grid:
        ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    
    plt.savefig(f'{filename}.eps', format='eps', bbox_inches='tight')
    plt.show()
    plt.close()


dg_columns = [0, 3, 11, 19, 27, 35, 43]
create_plot(dg_columns, 'ΔG (kJ/mol)', 'delta_g_plot', 'Gibbs Free Energy vs Temperature')


logk_columns = [0, 6, 14, 22, 30, 38, 46]
create_plot(logk_columns, 'log K', 'log_k_plot', 'Log K vs Temperature')


dh_columns = [0, 1, 9, 17, 25, 33, 41]  
create_plot(dh_columns, 'ΔH (kJ/mol)', 'delta_h_plot', 'Enthalpy vs Temperature', legend_outside=True)


ds_columns = [0, 2, 10, 18, 26, 34, 42]  
create_plot(ds_columns, 'ΔS (J/mol·K)', 'delta_s_plot', 'Entropy vs Temperature', legend_outside=True)