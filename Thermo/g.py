import numpy as np
import pandas as pd
from itertools import product


ranges = {
    'T': (50, 500, 25),
    'P': (1, 30, 1),
    'fco2andcdo': (0.25, 2.0, 0.25),
    'fh2': (0.25, 2.0, 0.25)
}


levels = {}
for key, (min_val, max_val, delta) in ranges.items():
    levels[key] = np.arange(min_val, max_val + delta / 2, delta)  


full_factorial = list(product(levels['T'], levels['P'], levels['fco2andcdo'], levels['fh2']))


doe_df = pd.DataFrame(full_factorial, columns=['T', 'P', 'fco2andcdo', 'fh2'])


doe_df.to_csv('full_factorial_doe.csv', index=False)

print("Full factorial DOE with 4900 samples generated and saved to 'full_factorial_doe.csv'")
print(f"Total runs: {len(doe_df)}")
print(doe_df.head())