import os

import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="NER Project Demo", page_icon="🧠", layout="centered")

MODEL_PATH = "./bert_ner_final"

LABEL_MAPPING = {
    "per": "Person",
    "geo": "Location",
    "gpe": "Location",
    "org": "Organization",
    "tim": "Time",
    "art": "Artifact",
    "eve": "Event",
    "nat": "Nature",
}

COLOR_MAP = {
    "geo": "#bae6fd",
    "gpe": "#bae6fd",
    "per": "#fecaca",
    "org": "#bbf7d0",
    "tim": "#fef08a",
    "art": "#e9d5ff",
    "eve": "#fed7aa",
    "nat": "#e5e7eb",
}


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Ошибка: Не найдена папка '{MODEL_PATH}'")
        return None
    try:
        return pipeline(
            "ner",
            model=MODEL_PATH,
            tokenizer=MODEL_PATH,
            aggregation_strategy="simple",
            device=-1,
        )
    except Exception as e:
        st.error(f"Ошибка: {e}")
        return None


nlp = load_model()


def render_ner_html(text, entities):
    # Исправление 1: Разбиваем длинную строку с помощью скобок
    html_content = (
        '<div style="line-height: 3.5; font-family: sans-serif; '
        'font-size: 16px; margin-bottom: 3rem;">'
    )
    last_idx = 0

    entities = sorted(entities, key=lambda x: x["start"])

    for entity in entities:
        start, end = entity["start"], entity["end"]
        raw_label = entity["entity_group"]
        word = text[start:end]

        readable_label = LABEL_MAPPING.get(raw_label.lower(), raw_label.upper())
        color = COLOR_MAP.get(raw_label, "#e5e7eb")

        if start > last_idx:
            html_content += f"<span>{text[last_idx:start]}</span>"

        # Исправление 2: Добавляем # noqa: E501, чтобы flake8 игнорировал длину строк в HTML шаблоне
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
        """  # noqa: E501
        html_content += entity_html
        last_idx = end

    if last_idx < len(text):
        html_content += f"<span>{text[last_idx:]}</span>"

    html_content += "</div>"
    return html_content


st.title("🔍 NER: Анализ текста")
st.markdown("Система распознавания именованных сущностей (BERT)")

default_text = (
    "Steve Jobs presented the new iPhone in San Francisco at the Apple headquarters."
)
text = st.text_area("Введите текст:", default_text, height=100)

if st.button("Анализировать", type="primary"):
    # ВАЖНО: Убедитесь, что переменная nlp инициализирована где-то в коде
    if "nlp" in globals() and text:
        with st.spinner("Анализируем..."):
            results = nlp(text)  # noqa: F821
            html_result = render_ner_html(text, results)

            st.markdown("### Результат:")
            st.markdown(html_result, unsafe_allow_html=True)
            st.write("")

            with st.expander("Техническая информация (JSON)"):
                st.json(results)
