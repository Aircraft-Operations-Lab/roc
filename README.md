# roc: Robust Optimal Control for Flight Planning
      
## What is roc?

The Python Library roc is a software package developed by UC3M. It is a tool for robust flight planning considering the concept of free-routing airspace. The main features of roc are: 1) integrates horizontal and vertical decision-making, 2) incorporates uncertainty in meteorological variables. In addition, roc is very generic in terms of flight planning objectives. For instance, it can be used to optimize aircraft trajectory w.r.t minimum flight time, minimum fuel burn, minimum operating cost, and minimum climate impact. 

![alt text](test/Key_Figure.png)

**License:** roc is released under GNU Lesser General Public License v3.0 (LGPLv3). 

**Support:** Support of all general technical questions on roc, i.e., installation, application, and development, will be provided by Abolfazl Simorgh (abolfazl.simorgh@uc3m.es). 

Copyright (C) 2022, Universidad Carlos III de Madrid

## Citation info

**Ref 1:** Simorgh, Soler, M., Dietmuller, S., Matthes, S., Yamashita, H., Castino, H., Yin, F., Robust 4D Climate-Optimal Aircraft Trajectory Planning under Weather-Induced Uncertainties: Free-Routing Airspace. Transportation Research Part D (under review).

## How to run the library
The installation is the first step to start working with roc. In the following, the steps required to install the library are provided.

0. It is highly recommended to create a virtual environment (e.g., roc):
```python
conda create -n env_roc python=3.9
conda activate env_roc
```

1. Clone or download the repository. The roost source code is available on a public GitHub repository: https://github.com/Aircraft-Operations-Lab/roc. The easiest way to obtain it is to clone the repository using git: git clone https://github.com/Aircraft-Operations-Lab/roc.git.

2. Locate yourself in the roost (library folder) path, and run the following line, using terminal (MacOS and Linux) or cmd (Windows), which will install all dependencies:
```python
python setup.py install
```
it will install all the required dependencies.

## How to use it
There is a script in the test folder of the library, *test.py*, which provides a sample to get started with the library. This sample file contains comments explaining the required inputs, problem configurations, objective function selection (which includes flight planning objectives), optimization configurations, running, and output files. Notice that we use BADA4.2 to represent the aerodynamic and propulsive performance of the aircraft. Due to restrictions imposed by the BADA license, the current version on GitHub is incomplete, as two Python scripts related to the used aircraft performance model have been excluded (i.e., bada4.py and apm.py). Therefore, users need to define and input an aircraft performance model to the rfp.py module to work with the current library version. We are currently assessing the existing open-source aircraft performance models in order to make the complete library available to the public. 