Train models----
python src/train_model.py       # Binary PHI detector
python src/train_ner.py         # Generate spaCy NER training data
python -m spacy init config config.cfg --pipeline ner --lang en
python -m spacy train config.cfg --output ./models/phi_ner --paths.train ./training_data/train.spacy --paths.dev ./training_data/train.spacy

Run analysis on sample documents
python src/main.py
