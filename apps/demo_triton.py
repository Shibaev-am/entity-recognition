import streamlit as st
import numpy as np
import requests
from transformers import BertTokenizerFast
import torch
import os

st.set_page_config(page_title="NER Project Demo (Triton)", page_icon="🧠", layout="centered")

# === Конфигурация ===
TRITON_URL = "http://localhost:8000/v2/models/bert_ner/infer"
MODEL_NAME = "bert_ner"  # Имя модели в Triton
TOKENIZER_NAME = "bert-base-cased" # Или путь к локальной модели
TAG2IDX_PATH = "models/tag2idx.pt" # Путь к словарю тегов

LABEL_MAPPING = {
    "per": "Person",
    "geo": "Location",
    "gpe": "Location",
    "org": "Organization",
    "tim": "Time",
    "art": "Artifact",
    "eve": "Event",
    "nat": "Nature"
}

COLOR_MAP = {
    "geo": "#bae6fd", "gpe": "#bae6fd",
    "per": "#fecaca",
    "org": "#bbf7d0",
    "tim": "#fef08a",
    "art": "#e9d5ff", "eve": "#fed7aa",
    "nat": "#e5e7eb"
}

@st.cache_resource
def load_resources():
    try:
        tokenizer = BertTokenizerFast.from_pretrained(TOKENIZER_NAME)
        
        if os.path.exists(TAG2IDX_PATH):
            tag2idx = torch.load(TAG2IDX_PATH)
            idx2tag = {v: k for k, v in tag2idx.items()}
        else:
            st.error(f"Файл словаря {TAG2IDX_PATH} не найден!")
            return None, None

        return tokenizer, idx2tag
    except Exception as e:
        st.error(f"Ошибка загрузки ресурсов: {e}")
        return None, None

tokenizer, idx2tag = load_resources()

def query_triton(text, tokenizer):
    inputs = tokenizer(
        text,
        return_tensors="np",
        padding="max_length",
        truncation=True,
        max_length=128,
        return_offsets_mapping=True
    )
    
    input_ids = inputs["input_ids"].astype(np.int64)
    attention_mask = inputs["attention_mask"].astype(np.int64)
    offset_mapping = inputs["offset_mapping"][0] # [Seq, 2]

    payload = {
        "inputs": [
            {
                "name": "input_ids",
                "shape": input_ids.shape,
                "datatype": "INT64",
                "data": input_ids.tolist()
            },
            {
                "name": "attention_mask",
                "shape": attention_mask.shape,
                "datatype": "INT64",
                "data": attention_mask.tolist()
            }
        ],
        "outputs": [
            {
                "name": "logits"
            }
        ]
    }

    try:
        response = requests.post(TRITON_URL, json=payload)
        response.raise_for_status()
        result_data = response.json()
        
        # Формат Triton JSON response: {"outputs": [{"name": "logits", "data": [...], "shape": [...]}]}
        logits_data = result_data["outputs"][0]["data"]
        shape = result_data["outputs"][0]["shape"]
        
        logits = np.array(logits_data).reshape(shape)
        preds = np.argmax(logits, axis=2)[0] # [Seq]
        
    except Exception as e:
        st.error(f"Ошибка Triton Inference: {e}")
        return []

    entities = []
    current_entity = None

    for idx, (pred_idx, offset) in enumerate(zip(preds, offset_mapping)):
        start, end = offset
        if start == end: continue
        
        tag = idx2tag.get(pred_idx, "O")
        
        if tag.startswith("B-"):
            if current_entity:
                entities.append(current_entity)
            current_entity = {
                "entity_group": tag.split("-")[1],
                "start": int(start),
                "end": int(end),
                "score": 1.0
            }
        elif tag.startswith("I-") and current_entity:
            type_ = tag.split("-")[1]
            if type_ == current_entity["entity_group"]:
                current_entity["end"] = int(end)
            else:
                entities.append(current_entity)
                current_entity = None
        else:
            if current_entity:
                entities.append(current_entity)
                current_entity = None
                
    if current_entity:
        entities.append(current_entity)
        
    return entities

def render_ner_html(text, entities):
    html_content = '<div style="line-height: 3.5; font-family: sans-serif; font-size: 16px; margin-bottom: 3rem;">'
    last_idx = 0
    
    entities = sorted(entities, key=lambda x: x['start'])
    
    for entity in entities:
        start, end = entity['start'], entity['end']
        raw_label = entity['entity_group']
        word = text[start:end]
        
        readable_label = LABEL_MAPPING.get(raw_label.lower(), raw_label.upper())
        color = COLOR_MAP.get(raw_label, "#e5e7eb")
        
        if start > last_idx:
            html_content += f'<span>{text[last_idx:start]}</span>'
        
        entity_html = f"""
        <span style="display: inline-block; position: relative; line-height: 1.0; vertical-align: baseline; margin: 0 4px;">
            <span style="
                background-color: {color}; 
                color: #111827; 
                padding: 4px 6px; 
                border-radius: 6px; 
                font-weight: 500;
                border: 1px solid rgba(0,0,0,0.1);">
                {word}
            </span>
            <span style="
                position: absolute;
                top: 100%;
                left: 50%;
                transform: translateX(-50%);
                font-size: 0.75em;
                color: {color}; 
                margin-top: 0.5rem;
                font-weight: 600;
                white-space: nowrap;
                pointer-events: none;
                opacity: 0.9;">
                {readable_label}
            </span>
        </span>
        """
        html_content += entity_html
        last_idx = end
        
    if last_idx < len(text):
        html_content += f'<span>{text[last_idx:]}</span>'
        
    html_content += '</div>'
    return html_content

st.title("🔍 NER: Анализ текста (Triton Inference)")
st.markdown(f"Сервер: `{TRITON_URL}` | Модель: `{MODEL_NAME}`")

default_text = "Steve Jobs presented the new iPhone in San Francisco at the Apple headquarters."
text = st.text_area("Введите текст:", default_text, height=100)

if st.button("Анализировать", type="primary"):
    if tokenizer and text:
        with st.spinner("Запрос к Triton Server..."):
            results = query_triton(text, tokenizer)
            html_result = render_ner_html(text, results)
            
            st.markdown("### Результат:")
            st.markdown(html_result, unsafe_allow_html=True)
            st.write("")
            
            with st.expander("Техническая информация (JSON)"):
                st.json(results)
    else:
        st.error("Не удалось инициализировать компоненты (Triton/Tokenizer). Проверьте логи.")

