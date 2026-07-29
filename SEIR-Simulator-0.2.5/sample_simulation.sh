#!/usr/bin/env bash

# example running seir_simulator
#./seir_simulator folder=./dec_15_EPI iu=0.92 fu=0.70 rd=15 R0=10 ts=7 force=seasonal T=140 runs=1000

#To do: 
#./seir_simulator folder=./ iu=0.92 fu=0.70 rd=15 R0=10 ts=7 force=term T=140 runs=1


#eta test
./seir_simulator_gamma folder=./ ts=7 force=term T=140 runs=20 \
eta_i=1 eta_f=1 R0_i=0 R0_f=1 R0_rs=1 R0_rd=140 v_i=0.0 v_f=0.0 force=none

