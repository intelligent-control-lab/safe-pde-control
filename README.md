# safe-pde-control
This the released code for L4DC 2025 "Safe pde boundary control with neural operators" ([PDF](https://arxiv.org/pdf/2411.15643)). 

## Preparation
The code is based on [Julia](https://julialang.org/) and Python. It is tested with Julia v1.9.4. Check [here](https://julialang.org/downloads/oldreleases/) to install Julia environment. Install `NeuralOperators.jl` from [here](https://docs.sciml.ai/NeuralOperators/stable/#Installation). For Python part, the code is based on `PDEControlGym`, which can be installed based on [doc](https://pdecontrolgym.readthedocs.io/en/latest/guide/install.html).

## Data collection
Train or download the PPO and SAC models for all the environments following `README.md` of [PDEControlGym](https://github.com/lukebhan/PDEControlGym).
To collect data for hyperbolic environment, see Jupyter file `HyperbolicPDEExample.ipynb` for details. To collect data for parabolic environment, see Jupyter file `ParabolicPDEExample.ipynb` for details. To collect data for Navier-Stokes environment, see Jupyter file `NS2DExample.ipynb` for details.

## Data preprocessing
Check out `preprocess_hyperbolic.ipynb` for data preprocessing of collected hyperbolic PDE data. Check out `preprocess_parabolic.ipynb` for data preprocessing of collected parabolic PDE data. Check out `preprocess_ns.ipynb` for data preprocessing of collected Navier-Stokes data. 

## Model training 
Under hyperbolic equation, see `train_hyper_all_pf.jl` for neural operator training and the Jupyter file `train_cbf_hyper.ipynb` for neual BCBF training. Similarly, see `train_para_all_pf.jl` to train neural operator and the Jupyter file `train_cbf_parabolic.ipynb` for neual BCBF training under parabolic equation. see `train_ns_all_pf.jl` to train neural operator and the Jupyter file `train_cbf_ns.ipynb` for neual BCBF training under Navier-Stokes equation. 


## Evaluation of online safety filtering
For the safety filtering over the collected trajectories, see Jupyter file `test_cbf_hyper.ipynb` for hyperbolic equation, `test_cbf_parabolic.ipynb` for parabolic equation, and `test_cbf_ns.ipynb` for Navier-Stokes equation. For the reward and PF metric metric evaluation, see Jupyter file `transportPDE/HyperbolicPDEExample.ipynb` for hyperbolic equation,  Jupyter file `reactionDiffusionPDE/ParabolicPDEExample.ipynb` for parabolic equation and Jupyter file `NS2Dtest.ipynb` Navier-Stokes equation.
