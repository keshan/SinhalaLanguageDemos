import streamlit as st

from googletrans import Translator
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
# from huggingface_hub import snapshot_download

page = st.sidebar.selectbox("Model ", ["Finetuned on News data", "Pretrained GPT2"])
translator = Translator()

def load_model(model_name):
    with st.spinner('Waiting for the model to load.....'):
        # snapshot_download('flax-community/Sinhala-gpt2')
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, pad_token_id=tokenizer.eos_token_id)
    st.success('Model loaded!!')
    return model, tokenizer

seed = st.sidebar.text_input('Starting text', 'ආයුබෝවන්')
seq_num = st.sidebar.number_input('Number of sequences to generate ', 1, 20, 5)
max_len = st.sidebar.number_input('Length of a sequence ', 5, 300, 100)
gen_bt = st.sidebar.button('Generate')

def generate(model, tokenizer, seed, seq_num, max_len):
    sentences = []
    input_ids = tokenizer.encode(seed, return_tensors='pt')
    beam_outputs = model.generate(
        input_ids, 
        do_sample=True, 
        max_length=max_len, 
        top_k=50, 
        top_p=0.95, 
        temperature=0.7,
        num_return_sequences=seq_num,
        no_repeat_ngram_size=2,
        early_stopping=True
    )
    for beam_out in beam_outputs:
        sentences.append(tokenizer.decode(beam_out, skip_special_tokens=True))
    return sentences
    
if page == 'Pretrained GPT2':
    st.title('Sinhala Text generation with GPT2')
    st.markdown('A simple demo using Sinhala-gpt2 model trained during hf-flax week')

    model, tokenizer = load_model('flax-community/Sinhala-gpt2')


    if gen_bt:
        try:
            with st.spinner('Generating...'):
                # generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
                # seqs = generator(seed, max_length=max_len, num_return_sequences=seq_num)
                seqs = generate(model, tokenizer, seed, seq_num, max_len)
            for i, seq in enumerate(seqs):
                st.info(f'Generated sequence {i+1}:')
                st.write(seq)
                st.info(f'English translation (by Google Translation):')
                st.write(translator.translate(seq, src='si', dest='en').text)
        except Exception as e:
            st.exception(f'Exception: {e}')
else:
    
    st.title('Sinhala Text generation with Finetuned GPT2')
    st.markdown('This model has been finetuned Sinhala-gpt2 model with 6000 news articles(~12MB)')
    
    model, tokenizer = load_model('keshan/sinhala-gpt2-newswire')


    if gen_bt:
        try:
            with st.spinner('Generating...'):
                # generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
                # seqs = generator(seed, max_length=max_len, num_return_sequences=seq_num)
                seqs = generate(model, tokenizer, seed, seq_num, max_len)
            for i, seq in enumerate(seqs):
                st.info(f'Generated sequence {i+1}:')
                st.write(seq)
                st.info(f'English translation (by Google Translation):')
                st.write(translator.translate(seq, src='si', dest='en').text)
        except Exception as e:
            st.exception(f'Exception: {e}')
st.markdown('____________')
