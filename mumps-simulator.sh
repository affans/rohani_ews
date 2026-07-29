#!/usr/bin/env bash

#./SEIR-Simulator-0.1.0/seir_simulator folder=./data/ \
# iu=0.92 fu=0.70 rd=15 R0=10 ts=7 force=seasonal T=140 runs=1 rho=0.0769 gamma=0.1667 sa=0.3


./SEIR-Simulator-0.2.0/seir_simulator_gamma folder=./data/ \
 v_i=0.92 v_f=0.70 v_rd=15 R0_i=10 R0_f=10 ts=7 force=seasonal T=140 runs=1 rho=0.0769 gamma=0.1667 sa=0.3
 
# Run workflow: do LHS, do cross-validation, calculate EWS, save optimum weights and thresholds 
PYTHONPATH=~/ecology/finches/ python3.7 ./py/workflow.py 

