# Vi Laya intent classification demo

This app loads the trained checkpoint already present in this workspace:
`../vi_laya_checkpoint/vi-laya-s0/`. It uses the checkpoint's saved tokenizer,
encoder, weights, questions and calibration configuration. It does not download a
replacement model.

## Audit findings

- The model artifact is the Laya RL-agent checkpoint layout: `model.safetensors`,
  `encoder/`, `tokenizer/`, `rl_agent_config.json`, `questions.json`, and
  `train_report.json`. The root checkpoint and `checkpoint_latest/` each contain
  a weights file; both are retained by the package script.
- Laya's legacy Python API documents `laya.load(local_checkpoint_dir)` and
  `agent.predict(state, questions)`. The app pins `laya==0.3.25`, from the
  local-checkpoint API line before the 0.5.x default-engine change. The checkpoint
  itself does not record the exact Laya package version used to train it, and the
  training implementation is not included in this workspace, so exact training
  version provenance cannot be established here.
- `rl_agent_config.json` sets choice temperature to 5.0; the Laya loader reads
  this calibration from the checkpoint. It also records a held-out evaluation
  accuracy of 0.8315 after calibration over 540 items and notes that the fitted
  choice temperature reached its upper clamp. Those are training report values,
  not a new evaluation of this demo.
- The trained checkpoint's `questions.json` has 15 family choices and one flat,
  shared 14-option `intent` question. The dataset beside `families.json` provides
  the complete hierarchical taxonomy: 15 families and 125 intents (the files
  contain 125, not 130).
- `config/families.json` and `config/intents.json` are copies of those dataset
  catalogs. Their 125 family assignments and `option_text` descriptions match
  every option in the Stage 2 train, dev and test records. The older
  `config/questions.example.json` remains a small illustrative sample and is not
  used for inference.
- The predictor combines the checkpoint's family-question instructions with the
  canonical family descriptions and builds
  `intents: {family_name: {type, instructions, criteria}}` from the catalogs.
  Stage 2 sends only the selected family's choices to Laya, not all 125 at once.
  The original checkpoint files and weights are not modified.
- The saved schema treats `out_of_scope` as a family. For that result the app
  skips Stage 2, returns `out_of_scope` as the intent, uses intent confidence 1.0,
  and sets combined confidence to family confidence.

## Requirements

- Ubuntu with Python 3.10 or newer. Python 3.10 is available in the current
  workspace machine; the system `python` alias may be absent, so the commands
  below use `python3.10`.
- CPU inference is supported. The checkpoint has 322M parameters; cloud memory
  limits may prevent it from loading on smaller instances.

## Install on Ubuntu

Run these commands from `vi_laya_streamlit_demo/`:

```bash
sudo apt update
sudo apt install -y python3.10-venv
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -c 'import laya, streamlit; print("Laya", getattr(laya, "__version__", "version unavailable")); print("Streamlit", streamlit.__version__)'
```

The requirements pin the tested Laya, Transformers, Streamlit and CPU-only
PyTorch versions. For CUDA, create a separate requirements variant with a PyTorch
build matching your driver.

## Run

From `vi_laya_streamlit_demo/` with the virtual environment active:

```bash
streamlit run app.py --server.fileWatcherType none
```

Disabling the file watcher avoids optional Transformers image-module probes in
this text-only environment. Restart Streamlit after code changes.

The default checkpoint and taxonomy paths are resolved from this repository's
actual layout. To use another location, set `LAYA_MODEL_DIR`, `LAYA_QUESTIONS_FILE`,
`LAYA_FAMILIES_FILE` or `LAYA_INTENTS_FILE` as needed.

The sidebar threshold defaults to 0.80. It uses Laya's `answer_confidence`
(the selected choice's probability, `max(p)`), which is the quantity fitted by
temperature calibration. Laya's separate `confidence` field is normalized
entropy for choice questions and is retained in the expandable raw response; it
is not used for the combined score. The threshold is for demonstration only,
not an approved production policy. The app accepts English, Hindi and Hinglish
text and includes sample messages in those languages.

## Push to GitHub

Run Git from the workspace root (`laya-tuned/`), not from the app subdirectory.
The root `.gitignore` excludes the virtual environment, secrets, generated ZIP,
and duplicate `checkpoint_latest/` directory. The active 644 MB
`model.safetensors` must use Git LFS; `.gitattributes` is already configured for
all `.safetensors` files. The complete 1.2 GB ZIP is not needed for the app and
should not be pushed.

```bash
sudo apt update
sudo apt install -y git git-lfs
git lfs install
cd /path/to/laya-tuned
git init -b main
git lfs track '*.safetensors'
git add .
git lfs ls-files
git status --short
git commit -m "Prepare Vi Laya intent demo for deployment"
git remote add origin https://github.com/OWNER/REPOSITORY.git
git push -u origin main
```

Replace `OWNER/REPOSITORY` with the GitHub repository you created. Before
committing, confirm `git lfs ls-files` lists the checkpoint weights and
`git status` does not list `.venv`, secrets or `vi_laya_model_bundle.zip`.
GitHub Free permits LFS files up to 2 GB each and includes 10 GiB of LFS storage
and download bandwidth; the single active weights file is about 644 MB. See
[GitHub LFS limits and quotas](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage)
and [Git LFS billing](https://docs.github.com/en/billing/concepts/product-billing/git-lfs).

## Deploy on Streamlit Community Cloud

1. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/) with the
   GitHub account that can access the repository.
2. Choose **Create app**, select the repository and `main` branch, and set the
   app file path to `vi_laya_streamlit_demo/app.py`.
3. In **Advanced settings**, choose Python 3.12 and deploy. The app has no
   credentials or secrets to configure.
4. Open the build logs if startup fails. The app requires the Git LFS checkpoint
   object; Community Cloud supports Git LFS repositories. See the
   [Streamlit deployment guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)
   and [file organization guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/file-organization).

Community Cloud resource limits are approximate and can change; Streamlit's
current docs list a maximum around 2.7 GB memory. This 322M-parameter model may
approach that limit when loaded with the app. If it exceeds the app's memory,
use a host with more RAM or request a Community Cloud resource increase if
eligible. Monitor the app logs after the first deploy. See
[Community Cloud resource limits](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app#resource-limits).

## Package the model

From `vi_laya_streamlit_demo/`:

```bash
python package_model.py
```

The script writes `../vi_laya_model_bundle.zip`, preserves the source checkpoint
layout under `vi_laya_checkpoint/vi-laya-s0/`, reopens and CRC-checks the archive,
checks all source files and required model artifacts, then prints its size and
inventory. Use `--source` and `--output` to choose other paths. It never writes to
the checkpoint directory.

## Tests

`test_report.md` records the actual results of dependency, checkpoint, inference
and Streamlit startup checks run in this workspace. Checks that could not run due
to an unavailable dependency or execution constraint are marked **Blocked**, not
passed.
