# EIREstimator

## Overview

The excitation-inhibition ratio (EIR) is a key neurophysiological biomarker reflecting the dynamic balance between excitatory and inhibitory neural activities. Abnormal EIR changes are associated with neurological disorders. The neural mass model (NMM), capable of reproducing cortical dynamics with physiologically meaningful parameters, is previously used for estimation of EIR changes from electroencephalography (EEG). However, such inverse estimation requires multiple customized programming steps. To streamline this process and improve accessibility, we developed EIREstimator, a Python-based open-source software with graphical user interface (GUI) for estimating EIR changes from EEG using NMMs. We designed the GUI so that users can conduct a full analysis pipeline comprising data import, signal preprocessing, model parameter configuration, EIR dynamic estimation, and results visualization without coding. Two estimation algorithms were implemented, i.e., variational Bayesian noise-adaptive constrained ensemble Kalman filter (vbcEnKF) and particle swarm optimization (PSO). Using epileptic EEG as a case, we illustrated the EIREstimator’s workflow. Application to sleep EEG further confirmed its effectiveness in identifying stage-dependent EIR patterns. Evaluation with simulated data showed PSO was more robust to initial parameter uncertainty, while vbcEnKF achieved higher accuracy and stability under noise corruption. By incorporating two complementary algorithms within a standardized workflow, EIREstimator provides an accessible and integrated platform for NMM-based EIR estimation from EEG. Our software lowers the technical barrier for two algorithms, which facilitates reproducible analysis of EIR dynamics in both pathological and physiological states. 

## Features
Simulate neural mass models

Import experimental recordings or simulated signals

Graphical user interface (GUI)

## Installation
We recommend using Anaconda to deploy the Python environment for EIREstimator.

Set up conda environment by

    conda env create --file environment.yml
    conda activate EIREstimator

Then,run the nmm generator by main_devided.py

Run the signal simulator by Signal_simulator.py

Run the estimator by vbEnKF_simulator.py or PSO_simulator.py
