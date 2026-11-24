# Vi Laya Intent Classification Demo

This repository contains a Streamlit app for the Vi Laya two-stage intent
classifier. It predicts one of 15 intent families, then evaluates only the
intents assigned to that family. The dataset taxonomy contains 125 intents.

**Live app:** [Open the Vi Laya classifier](https://laya-intent-classifier-j3lwdmfy6hucbg4lerek3m.streamlit.app/)

## Run locally

```bash
cd vi_laya_streamlit_demo
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py --server.fileWatcherType none
```

The checkpoint is under `vi_laya_checkpoint/vi-laya-s0/`. See the
[application README](vi_laya_streamlit_demo/README.md) for the audit, setup,
Git LFS instructions and Streamlit Community Cloud deployment guide.

## Model files and GitHub

The active `model.safetensors` file is about 644 MB and is tracked with Git LFS.
The generated `vi_laya_model_bundle.zip` and duplicate `checkpoint_latest/`
checkpoint are excluded from Git. GitHub's LFS storage and bandwidth quotas may
apply when deploying or downloading the model; details are in the application
README.

## Demo status

The default confidence threshold is 0.80 for demonstration only. It is not an
approved production routing policy. Review the documented test report and
taxonomy limitations before using the app with live traffic.
# laya-intent-classifier
