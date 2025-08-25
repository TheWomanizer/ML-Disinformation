
# Script para cargar modelos BERT en el ensemble
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def load_bert_model(model_name):
    model_path = f'../models/bert_models/{model_name}_final'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    return tokenizer, model

# Mejor modelo: roberta
# F1-Score: 0.8808
