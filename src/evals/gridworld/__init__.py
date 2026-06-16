"""GridWorld recall eval (D-032).

Predictor-agnostic metric core (`recall.py`) + closed-form frame readout (`readout.py`).
The model-driving Eval-interface adapter (score/report via encode->rollout->decode) is added once
a GridWorld-trained tokenizer+dynamics exists; until then the core is validated against synthetic
baselines (oracle / copy-last / chance) — see src/tests/test_gridworld_eval.py.
"""
