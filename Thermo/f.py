import numpy as np
import pandas as pd
from SALib.sample import saltelli


problem = {
    'num_vars': 4,
    'names': ['T', 'P', 'fco2andco', 'fh2'],
    'bounds': [[50, 500], [1, 30], [0.5, 2.0], [0.5, 2.0]]
}


increments = {
    'T': 50,
    'P': 1,
    'fco2andcdo': 0.25,
    'fh2': 0.5
}


N = 500  


samples = saltelli.sample(problem, N, calc_second_order=True)


scaled_samples = np.zeros_like(samples)
for i, name in enumerate(problem['names']):
    min_val, max_val = problem['bounds'][i]
    delta = increments[name]
    
    scaled_samples[:, i] = np.round(samples[:, i] / delta) * delta
    
    scaled_samples[:, i] = np.clip(scaled_samples[:, i], min_val, max_val)


doe_df = pd.DataFrame(scaled_samples, columns=problem['names'])


doe_df.to_csv('sobol_doe_1000.csv', index=False)

print("Sobol DOE with 1000 samples generated and saved to 'sobol_doe_1000.csv'")
print(doe_df.head())