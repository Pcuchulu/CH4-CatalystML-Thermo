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
    r'CH$_4$',
    r'H$_2$O',
    r'N$_2$',
    r'CO$_2$',
    r'CO',
    r'H$_2$',
    r'C'
]


markers = ['o', 's', '^', 'D', 'v', 'p', '<']









legend_fontsize = 9  
legend_loc = 'upper center'  
legend_bbox = (0.5, 0.98)  
legend_ncol = len(legends)  
legend_frameon = True  
legend_edgecolor = 'black'  


file_path = 'Thermodynamics data.xlsx'
sheet_name = 'f and v'


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


y_cols = [13, 14, 15, 16, 17, 18, 19]


create_plot(x_col=3, y_cols=y_cols, rows=[2, 17], x_label='Flow rate H$_2$ (m$_3$/h)', 
            y_label='Weight fraction (%)', filename='plot_1', title='Plot 1', legend_outside=True)


create_plot(x_col=3, y_cols=y_cols, rows=[19, 34], x_label='Flow rate H$_2$ (m$_3$/h)', 
            y_label='Weight fraction (%)', filename='plot_2', title='Plot 2', legend_outside=True)


create_plot(x_col=3, y_cols=y_cols, rows=[36, 51], x_label='Flow rate H$_2$ (m$_3$/h)', 
            y_label='Weight fraction (%)', filename='plot_3', title='Plot 3', legend_outside=True)


create_plot(x_col=3, y_cols=y_cols, rows=[53, 68], x_label='Flow rate H$_2$ (m$_3$/h)', 
            y_label='Weight fraction (%)', filename='plot_4', title='Plot 4', legend_outside=True)


create_plot(x_col=4, y_cols=y_cols, rows=[70, 88], x_label='Reactor Temperature (C)', 
            y_label='Weight fraction (%)', filename='plot_6', title='Plot 6', legend_outside=True)


create_plot(x_col=0, y_cols=y_cols, rows=[90, 109], x_label='Reactor Pressure (bar)', 
            y_label='Weight fraction (%)', filename='plot_7', title='Plot 7', legend_outside=True)


create_plot(x_col=0, y_cols=y_cols, rows=[111, 130], x_label='Reactor Pressure (bar)', 
            y_label='Weight fraction (%)', filename='plot_8', title='Plot 8', legend_outside=True)


create_plot(x_col=0, y_cols=y_cols, rows=[132, 151], x_label='Reactor Pressure (bar)', 
            y_label='Weight fraction (%)', filename='plot_9', title='Plot 9', legend_outside=True)


create_plot(x_col=0, y_cols=y_cols, rows=[153, 172], x_label='Reactor Pressure (bar)', 
            y_label='Weight fraction (%)', filename='plot_10', title='Plot 10', legend_outside=True)


create_plot(x_col=2, y_cols=y_cols, rows=[174, 189], x_label='Flow rate CO & CO$_2$ (m$_3$/h)', 
            y_label='Weight fraction (%)', filename='plot_11', title='Plot 11', legend_outside=True)


create_plot(x_col=2, y_cols=y_cols, rows=[191, 206], x_label='Flow rate CO & CO$_2$ (m$_3$/h)', 
            y_label='Weight fraction (%)', filename='plot_12', title='Plot 12', legend_outside=True)


create_plot(x_col=2, y_cols=y_cols, rows=[208, 223], x_label='Flow rate CO & CO$_2$ (m$_3$/h)', 
            y_label='Weight fraction (%)', filename='plot_13', title='Plot 13', legend_outside=True)


create_plot(x_col=2, y_cols=y_cols, rows=[225, 240], x_label='Flow rate CO & CO$_2$ (m$_3$/h)', 
            y_label='Weight fraction (%)', filename='plot_14', title='Plot 14', legend_outside=True)