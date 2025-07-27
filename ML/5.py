import os
import numpy as np


os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


try:
    from ase.build import fcc111, add_adsorbate, molecule
    from fairchem.core import OCPCalculator
    from ase.optimize import LBFGS
    import torch
    import torch_sparse
except ImportError as e:
    print(f"Import error: {e}")
    print("Ensure fairchem-core, ase, torch, torch-scatter, and torch-sparse are installed.")
    print("Install torch-sparse with: pip install torch-sparse -f https://data.pyg.org/whl/torch-<version>+<cuda>.html")
    exit(1)

import warnings


warnings.filterwarnings("ignore", category=DeprecationWarning)


material = "Pd"  
surface_type = "fcc111"  
ads_site = "fcc"  
checkpoint_path = "C:\\Users\\pcu\\Desktop\\eq2_153M_ec4_allmd.pt"  


try:
    calc = OCPCalculator(
        checkpoint_path=checkpoint_path,
        cpu=not torch.cuda.is_available()  
    )
except Exception as e:
    print(f"Error loading model: {e}")
    print("Verify the checkpoint path: C:\\Users\\pcu\\Desktop\\eq2_153M_ec4_allmd.pt")
    exit(1)


co = molecule("CO")
co.calc = calc
try:
    co_energy = co.get_potential_energy()
    print(f"Gas-phase CO energy: {co_energy:.3f} eV")
except Exception as e:
    print(f"Error calculating CO energy: {e}")
    exit(1)


slab = fcc111(material, size=(3, 3, 3), vacuum=8.0)
slab.calc = calc
try:
    dyn = LBFGS(slab)
    dyn.run(fmax=0.05, steps=100)
    slab_energy = slab.get_potential_energy()
    print(f"Clean slab energy ({material} {surface_type}): {slab_energy:.3f} eV")
except Exception as e:
    print(f"Error calculating slab energy: {e}")
    exit(1)


slab_co = slab.copy()
adsorbate = molecule("CO")
add_adsorbate(slab_co, adsorbate, height=2.0, position=ads_site)
slab_co.calc = calc
try:
    dyn = LBFGS(slab_co)
    dyn.run(fmax=0.05, steps=100)
    slab_co_energy = slab_co.get_potential_energy()
    print(f"Slab+CO energy: {slab_co_energy:.3f} eV")
except Exception as e:
    print(f"Error calculating slab+CO energy: {e}")
    exit(1)


adsorption_energy = slab_co_energy - (slab_energy + co_energy)
print(f"Predicted adsorption energy for CO on {material} ({surface_type}, {ads_site} site): {adsorption_energy:.3f} eV")