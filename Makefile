# 简易工作流
.PHONY: mfe rbp fig2 fig3 all

PY=python

mfe:
	$(PY) scripts/calc_mfe.py --input data/raw/predict_eval.csv --output data/derived/mfe_predict.csv
	$(PY) scripts/calc_mfe.py --input data/raw/generated_topk.csv --output data/derived/mfe_generated.csv

rbp:
	$(PY) scripts/scan_rbp_fimo.py --input data/raw/predict_eval.csv --output data/derived/rbp_predict.csv
	$(PY) scripts/scan_rbp_fimo.py --input data/raw/generated_topk.csv --output data/derived/rbp_generated.csv

fig2:
	$(PY) scripts/eval_predictor.py --pred data/raw/predict_eval.csv --mfe data/derived/mfe_predict.csv --rbp data/derived/rbp_predict.csv --outdir figs/outputs
	$(PY) scripts/plot_fig2_predictor.py --input data/raw/predict_eval.csv --outdir figs/outputs

fig3:
	$(PY) scripts/plot_fig3_generation.py --gen data/raw/generated_topk.csv --mfe data/derived/mfe_generated.csv --rbp data/derived/rbp_generated.csv --outdir figs/outputs

all: mfe rbp fig2 fig3
