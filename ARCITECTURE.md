optimizer-comparison/
├── README.md
├── pyproject.toml
├── requirements.txt
├── .gitignore
│
├── configs/
│   ├── config.yaml
│   ├── mode/
│   │   ├── mock.yaml
│   │   ├── smoke.yaml
│   │   └── full.yaml
│   ├── model/
│   │   ├── tiny.yaml
│   │   └── qwen_0_5b.yaml
│   ├── data/
│   │   └── openwebtext_100k.yaml
│   ├── optimizer/
│   │   ├── adamw.yaml
│   │   └── muon.yaml
│   ├── training/
│   │   └── default.yaml
│   ├── evaluation/
│   │   └── lm_eval.yaml
│   ├── tracking/
│   │   └── mlflow.yaml
│   └── experiment/
│       ├── mock_adamw.yaml
│       ├── smoke_adamw_tiny.yaml
│       ├── adamw_baseline.yaml
│       └── muon_baseline.yaml
│
├── src/
│   └── optimizer_comparison/
│       ├── train.py
│       ├── evaluate.py
│       ├── compare.py
│       ├── data/
│       ├── models/
│       ├── training/
│       ├── evaluation/
│       ├── tracking/
│       └── utils/
│
├── scripts/
│   ├── run_mock.sh
│   ├── run_smoke.sh
│   ├── run_adamw.sh
│   ├── run_muon.sh
│   └── run_eval.sh
│
├── notebooks/
├── outputs/
└── tests/
