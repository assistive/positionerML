# bert_mobile/
├── README.md
├── requirements.txt
├── config/
│   ├── model_config.yaml
│   ├── training_config.yaml
│   ├── mobile_config.yaml
│   └── vocab_config.yaml
├── src/
│   ├── __init__.py
│   ├── model_downloader.py
│   ├── vocab_builder.py
│   ├── data_processor.py
│   ├── bert_trainer.py
│   ├── mobile_converter.py
│   ├── tokenizer_utils.py
│   └── model_utils.py
├── scripts/
│   ├── download_model.py
│   ├── build_vocabulary.py
│   ├── prepare_data.py
│   ├── train_bert.py
│   ├── fine_tune_bert.py
│   ├── convert_to_mobile.py
│   ├── evaluate_model.py
│   └── deploy.py
├── data/
│   ├── raw/
│   ├── processed/
│   ├── vocabularies/
│   └── tokenized/
├── models/
│   ├── pretrained/
│   ├── trained/
│   ├── fine_tuned/
│   └── mobile/
├── notebooks/
│   ├── vocabulary_analysis.ipynb
│   ├── model_comparison.ipynb
│   └── mobile_optimization.ipynb
├── tests/
│   ├── test_vocabulary.py
│   ├── test_tokenizer.py
│   ├── test_training.py
│   └── test_mobile_conversion.py
└── deployment/
    ├── android/
    │   ├── app/
    │   └── integration_guide.md
    └── ios/
        ├── BertMobile/
        └── integration_guide.md
