"""Optional portfolio demo: python scripts/demo.py  (needs a trained checkpoint in checkpoints/ and `pip install gradio`)."""
import glob, torch, gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ckpt = sorted(glob.glob("checkpoints/distilbert_full_weighted_seed0/checkpoint-*"))[-1]
tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(ckpt).eval()
labels = model.config.id2label

def predict(text):
    with torch.no_grad():
        p = model(**tok(text, return_tensors="pt", truncation=True, max_length=64)).logits.softmax(-1)[0]
    return {labels[i]: float(p[i]) for i in range(len(labels))}

gr.Interface(predict, gr.Textbox(label="Tweet"), gr.Label(num_top_classes=6), title="Tweet emotion (DistilBERT)").launch()
