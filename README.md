404 Green Not Found - Green Lab Assignment 

Experiment Topic: Optimization Techniques for LLM Inferences on Mobile

This repository contains the runtable, results and scripts used while running and analyzing the experiment. 

Repository Structure
data/
This folder contains all datasets, prompts, and derived measurements used in the study.

  subset_plan/ – Includes the subset of rows extracted from the News QA Summarization dataset, organized by task. Each subset is accompanied by a text file defining the token threshold used to bucket summaries into long and short groups.

  runplan/ – Stores the complete experimental run plans. runplan_old.csv lists the initial 1,350 runs (150 tasks × 3 model families × 3 quantization levels), later reduced to 810 runs in runplan.csv after task pruning.

  prompts/ – Contains the task-specific prompts fed to each model, separated by model family.

  logs/ – Holds raw runtime logs captured during inference, stored as text files and grouped by model.

  analysis/ – Collects the processed measurements per research question, including statistical test results and outlier detection outputs.

src/
This folder hosts all scripts used to preprocess data, run model inferences, and generate analyses and figures.

  dataset/ – Scripts for dataset extraction, runplan generation, and prompt cleaning to ensure consistent formatting during shell execution.

  inference/ – Contains the main automation script orchestrator.sh managing inference and log collection. Includes utilities to detect failed runs and clean corresponding logs for re-execution.

  analysis/ – Organized by research question, this subfolder includes R scripts for generating analytical plots and statistical results visualization.

documentation/
This folder contains the final version of the research paper in PDF format.
