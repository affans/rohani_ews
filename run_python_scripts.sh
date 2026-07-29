#!/usr/bin/env bash


# SEIR simulator in ./SEIR-Simulator-0.2.5
# Run makefile to compile binaries 

# Use L1 regression to assign weights to EWS
python ./py/workflow.py

# Demonstration of workflow
python ./py/demonstration_figure.py

# Perform leave one out analysis on simulated dataset
python ./py/leave_one_out_analysis.py

# Generate plots from training data set
python ./py/training_figures.py

# Plot curvature of R0 for LHS samples
python .py/R0_plot.py

# Generate supplemental figures with covariates
python ./py/covar_figures.py

# Apply detection method to pertussis in the US
python ./py/pertussis_us.py

# Test performance of linear regression at detecting pertussis in the US
python ./py/pertussis_test.py

# Apply detection method to mumps in the UK
python ./py/mumps_england.py

# Apply detection method to plague and dengue examples
python ./py/plague.py

# Create introduction figure
python ./py/introduction_figure.py



