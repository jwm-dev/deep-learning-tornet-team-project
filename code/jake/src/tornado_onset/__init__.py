"""Tornado onset prediction from radar.

The question: given the radar scans observed so far, will a tornado appear in the next 5 and 10
minutes when none is visible yet? The model is the published TorNet detector's encoder
fine-tuned on that target with the authors' head on top; the frozen published detector is the
judge that sets the persistence floor and the ceiling. Model weights and data come from the
Hugging Face Hub. `run.py` at the repo root runs the steps.
"""
