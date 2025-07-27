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
    r'CO$_2$ Conversion',
    r'CO Conversion',
    r'CH$_4$ Yield',
    r'C Yield'
]


markers = ['o', 's', '^', 'D']









legend_fontsize = 9  
legend_loc = 'upper center'  
legend_bbox = (0.5, 0.98)  
legend_ncol = len(legends)  
legend_frameon = True  
legend_edgecolor = 'black'  


file_path = 'Thermodynamics data.xlsx'
sheet_name = 'Yield'


def create_plot(x_col, y_cols, rows, x_label, y_label, filename, title, legend_outside=True):
    
    data = pd.read_excel(file_path, sheet_name=sheet_name, usecols=[x_col] + y_cols, skiprows=rows[0]-1, nrows=rows[1]-rows[0]+1)
    
    data = data.apply(pd.to_numeric, errors='coerce')
    
    data = data.dropna(subset=data.columns[0])
    
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
    
    
    for i, col in enumerate(y_cols):
        
        if not data.iloc[:, i+1].isna().all():
            ax.plot(data.iloc[:, 0], data.iloc[:, i+1], marker=markers[i], 
                    linestyle='-', linewidth=1.8, markersize=7, label=legends[i], alpha=1)
    
    ax.set_xlabel(x_label, fontsize=14, fontfamily='Times New Roman', fontweight='bold')
    ax.set_ylabel(y_label, fontsize=14, fontfamily='Times New Roman', fontweight='bold')
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


y_cols = [15, 16, 17, 18]


create_plot(x_col=2, y_cols=y_cols, rows=[2, 20], x_label='Reactor Temperature (C)', 
            y_label='Convertion - Yield (%)', filename='plot_1', title='Plot 1', legend_outside=True)

create_plot(x_col=2, y_cols=y_cols, rows=[22, 40], x_label='Reactor Temperature (C)', 
            y_label='Convertion - Yield (%)', filename='plot_1', title='Plot 1', legend_outside=True)

create_plot(x_col=2, y_cols=y_cols, rows=[42, 60], x_label='Reactor Temperature (C)', 
            y_label='Convertion - Yield (%)', filename='plot_1', title='Plot 1', legend_outside=True)


create_plot(x_col=2, y_cols=y_cols, rows=[62, 80], x_label='Reactor Temperature (C)', 
            y_label='Convertion - Yield (%)', filename='plot_1', title='Plot 1', legend_outside=True)

create_plot(x_col=2, y_cols=y_cols, rows=[82, 100], x_label='Reactor Temperature (C)', 
            y_label='Convertion - Yield (%)', filename='plot_1', title='Plot 1', legend_outside=True)

create_plot(x_col=2, y_cols=y_cols, rows=[102, 120], x_label='Reactor Temperature (C)', 
            y_label='Convertion - Yield (%)', filename='plot_1', title='Plot 1', legend_outside=True)

create_plot(x_col=2, y_cols=y_cols, rows=[122, 140], x_label='Reactor Temperature (C)', 
            y_label='Convertion - Yield (%)', filename='plot_1', title='Plot 1', legend_outside=True)

create_plot(x_col=2, y_cols=y_cols, rows=[142, 160], x_label='Reactor Temperature (C)', 
            y_label='Convertion - Yield (%)', filename='plot_1', title='Plot 1', legend_outside=True)

create_plot(x_col=2, y_cols=y_cols, rows=[162, 180], x_label='Reactor Temperature (C)', 
            y_label='Convertion - Yield (%)', filename='plot_1', title='Plot 1', legend_outside=True)