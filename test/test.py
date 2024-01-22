import roc3
from roc3.weather import *
from importlib import reload
from roc3.rfp import *
from roc3.apm import *
from roc3.bada4 import *
from roc3.accf import *
import roc3
import roc3.pipelines
import pickle
import yaml


# Inputting the directory of roc library:
lib_directory = '/Users/abolfazlsimorgh/roc-climate/'
path_to_save = lib_directory + 'test/results/' + 'trj_EI{}.json'


# Input a dataset, which includes the required meteorological variables (a sample to prepare the pickle file can be found in the test folder)
with open(lib_directory + 'test/data/data_june27_r_aCCFs_10member.pickle', 'rb') as handle:
    wm = pickle.load(handle)  

# Input Aircraft Performance Datasets:
apm = BADA4_jet_CR('A330-341', full_path='Libraries/roc3/examples/A330-341.xml')


""" %%%%%%%%%%%%%%%%% CONFIGURATIONS %%%%%%%%%%%%%%%% """
# Default configurations
problem_setup = {
    'origin':  (43.4384, 5.2144), # Marseille  
    'destination': (53.3554, -2.2773), # Manchester 
    'h0': H2h(5000), 
    'hf': H2h(5000), 
    'CI': 1.0,
    'C_m': 0.51,
    'DP': 0.0,
    'm0': apm.MTOW*0.8,
    't0': 1526190000.0 + 0,
    'tas0': 150,
    'tasf': 150,
    'mfb':0,
    'lagr_2d': lambda *x: 0,
    'CP': 0.0,
    'ac_code': 'A330-341',
    'altitude': 'track',
    'airspeed': 'variable',
    'tas_slope_regularization': 1e3,
    'h_slope_regularization': 1.0,
    'clim_ig': False,
    'climate_impact': True,
    'EI': 0.0
    }

# Load  a configuration file to rewrite the default values
with open(lib_directory + "config-user.yml", "r") as ymlfile:
    confg = yaml.safe_load(ymlfile)
problem_setup.update (confg)   


""" Optimization configuration """ 
# list including weights of average climate impact in the cost functional
EI = [0.0, 20.0, 150.0, 300.0]

# weights of cost CI * (0.75 * time + C_m * fuel)
problem_setup['CI'] = 1.0

# weights of fuel consumption
problem_setup['C_m'] = 0.51

# Performing optimization for different values weighting ATR in the objective function
for indx, alp in enumerate (EI):
    problem_setup['EI'] = EI[indx]
    discretization_node = 80
    p1 = roc3.pipelines.Routing2DStandard(apm, wm.get_slice_with_n_member(10), problem_setup, discretization_node, solver_options={'linear_solver':'mumps'})
    trj = p1.solve(stop=False)
    with open(path_to_save.format(alp), 'w') as f:
        trj.save_to_json(f)   

