import streamlit as st
import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration

# Model yükleme işlevi
@st.cache_data
def load_model(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer, device

# Paraphrase işlevi
def paraphrase(text, model, tokenizer, device, max_len=512):
    model.eval()
    input_enc = tokenizer.encode_plus(text, return_tensors='pt', max_length=max_len, truncation=True)
    input_ids = input_enc['input_ids'].to(device)

    outputs = model.generate(input_ids, max_length=max_len, num_return_sequences=1, num_beams=5, temperature=0.0)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Streamlit uygulaması
def main():
    st.title("Türkçe Paraphrase Üretici - Transfer Learning")

    model_path = './model'
    model, tokenizer, device = load_model(model_path)

    text = st.text_area("Bir cümle girin:", height=100)
    if st.button("Paraphrase"):
        with st.spinner('Paraphrase üretiliyor...'):
            paraphrased_text = paraphrase(text, model, tokenizer, device)
            st.text_area("Üretilen Cümle:", value=paraphrased_text, height=100, key='paraphrased')

    if st.button("Önbelleği Temizle"):
        st.cache_data.clear()
        st.success("Önbellek temizlendi!")

if __name__ == '__main__':
    main()
