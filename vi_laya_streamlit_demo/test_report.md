# Vi Laya demo test report

Run date: 2026-11-24 (Asia/Kolkata)

Environment: Python 3.10.22, Laya 0.3.25, PyTorch 2.14.1+cpu,
Transformers 5.20.2, Streamlit 1.65.0. Inference used the original checkpoint
at `../vi_laya_checkpoint/vi-laya-s0/` and the family/intent catalogs copied from
`~/Downloads/laya_dataset/`.

| Check | Result | Evidence |
|---|---|---|
| Import and checkpoint load | PASS | Imported Laya, PyTorch, Transformers, Streamlit and predictor; loaded the local model, encoder and tokenizer. |
| Calibration | PASS | Loaded checkpoint temperatures `[5.0, 1.0, 1.0]`; Stage 2 uses the selected family's exact option set. |
| Catalog integrity | PASS | 15 families, 125 intents. Every intent is assigned once and defined once; all Stage 2 train/dev/test rows matched the family's option IDs, option descriptions and labels (0 mismatches). |
| Exact phrase: “I am unable to call since night.” | PASS | `network_issues` → `calls_not_working`; family confidence 0.866, intent confidence 0.935, combined 0.810; `AUTO_HANDLE`. |
| Outgoing-call wording | PASS | “I cannot make outgoing calls since last night.” → `network_issues` / `calls_not_working`; confidences 0.892 × 0.942 = 0.841. |
| Hinglish call wording | MIXED | “Mujhse kal raat se call nahi ho rahi hai.” → `call_features` / `call_waiting`; intent confidence 0.955. This remains a family-stage classification error for that wording. |
| Recharge wording | PASS | “I recharged yesterday but my balance has not been credited.” → `recharge_balance` / `recharge_done_but_not_credited`; confidences 0.903 × 0.909 = 0.820. |
| Family-specific option isolation | PASS | For each tested non-out-of-scope message, raw Stage 2 probability keys exactly matched the selected family's mapped intent keys. |
| Streamlit app and server | PASS | AppTest executed `app.py`, loaded the 15-family catalog and rendered the input/threshold controls with no exceptions. Streamlit serves on `127.0.0.1:8502`. |

## Catalog audit

`families.json` contains 15 families and 125 assigned intents (not 130).
`intents.json` contains exactly 125 intent definitions with their `option_text`.
Each Stage 2 training record uses only the options for its `family`; the option
IDs and descriptions match both catalogs across train (2,088 rows), dev (270),
and test (270). The predictor now builds its Stage 2 criteria from these catalogs
and sends only the predicted family's list to Laya.

The input “Mujhse kal raat se call nahi ho rahi hai” selected `call_features`,
which then correctly constrained Stage 2 to that family's five intents. The
catalog places the `calls_not_working` intent under `network_issues`; therefore a
family-stage error can still produce a wrong but family-valid Stage 2 choice.

The original checkpoint files and weights were not modified. The older
`questions.example.json` remains illustrative and is not used by the predictor.
