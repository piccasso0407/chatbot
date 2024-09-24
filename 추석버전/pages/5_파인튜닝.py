import streamlit as st
import PyPDF2
import docx  # For reading DOCX files
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain.schema import Document
from string import Template
import re
import os
import time
from langchain.prompts import ChatPromptTemplate
from funcs import load_css
import gc
import pandas as pd

# 페이지 설정: 화면을 가로로 넓게 사용
st.set_page_config(layout="wide")

from funcs import load_css, load_local_font
current_dir = os.path.dirname(os.path.abspath(__file__))
css_path = os.path.join(current_dir, 'style.css')

# CSS 파일 로드 함수
def load_css(file_name):
    if os.path.exists(file_name):
        with open(file_name, 'r', encoding='utf-8') as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        st.error(f"CSS 파일을 찾을 수 없습니다: {file_name}")

# CSS 파일 로드
load_css(css_path)

current_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(current_dir, "images", "lora.jpg", )


st.write("제 파인튜닝의 기본틀은 '필로소피 AI' 유튜브 채널을 참조했습니다.")
st.markdown('[필로소피AI 바로가기](https://youtu.be/QaOIcJDDDjo?si=oToxZutU-VzSGT5v)')

st.markdown("* * *")
st.subheader("|unsloth 설치")
st.markdown("unsloth 설치 코드는 아래와 같습니다. 리눅스 환경에서 실행해야 합니다.")
st.code('''
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes
        ''')
st.markdown("* * *")
st.subheader("|unsloth 모델 설정")

st.code('''
from unsloth import FastLanguageModel
import torch
max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/Meta-Llama-3.1-8B-bnb-4bit",      # Llama-3.1 15 trillion tokens model 2x faster!
    "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    "unsloth/Meta-Llama-3.1-70B-bnb-4bit",
    "unsloth/Meta-Llama-3.1-405B-bnb-4bit",    # We also uploaded 4bit for 405b!
    "unsloth/Mistral-Nemo-Base-2407-bnb-4bit", # New Mistral 12b 2x faster!
    "unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit",
    "unsloth/mistral-7b-v0.3-bnb-4bit",        # Mistral v3 2x faster!
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "unsloth/Phi-3-mini-4k-instruct",          # Phi-3 2x faster!d
    "unsloth/Phi-3-medium-4k-instruct",
    "unsloth/gemma-2-9b-bnb-4bit",
    "unsloth/gemma-2-27b-bnb-4bit",            # Gemma 2x faster!
] # More models at https://huggingface.co/unsloth

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Meta-Llama-3.1-8B",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)
        ''')

st.markdown("* * *")
st.subheader("|LoRA 어댑터 추가")

st.write("""이제 LoRA 어댑터를 추가하여 전체 매개변수의 1~10%만 업데이트하면 됩니다!
이 문장은 LoRA(Low-Rank Adaptation) 기술에 대해 설명하고 있습니다. 
         LoRA는 대규모 언어 모델을 미세 조정할 때 모든 매개변수를 업데이트하는 대신, 
         소수의 매개변수만을 업데이트하여 효율적으로 모델을 조정할 수 있게 해주는 기술입니다.
          이를 통해 계산 비용과 메모리 사용량을 크게 줄일 수 있으며, 모델의 성능을 유지하면서도 
         특정 작업에 맞게 빠르게 적응시킬 수 있습니다.""")


st.code('''
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

''')
st.image("images/lora.jpg")

st.markdown("* * *")
st.subheader("|데이터 준비")
st.write("""이제 yahma의 Alpaca 데이터셋을 사용합니다. 
        이는 원본 Alpaca 데이터셋에서 필터링된 52,000개의 데이터 포인트로 구성되어 있습니다. 
        이 코드 섹션을 여러분 자신의 데이터 준비 과정으로 대체할 수 있습니다.
        이 설명은 모델 훈련을 위한 데이터 준비 단계에 대해 언급하고 있습니다.
        Alpaca 데이터셋은 대화형 AI 모델을 훈련시키는 데 널리 사용되는 데이터셋입니다. 
        여기서는 원본 데이터셋의 일부를 필터링하여 사용하고 있으며, 
        사용자가 필요에 따라 자신의 데이터 준비 과정으로 이 부분을 대체할 수 있다고 안내하고 있습니다.""")

st.code('''alpaca_prompt = """아래는 작업을 설명하는 지시사항입니다. 입력된 내용을 바탕으로 적절한 응답을 작성하세요.
### 지시사항:
아래 입력에 대한 적절한 응답을 제공하세요.
### 입력:
{input}
### 응답:
{response}
"""

EOS_TOKEN = tokenizer.eos_token  # EOS_TOKEN 추가 필요

def formatting_prompts_func(examples):
    inputs = examples["input"]
    responses = examples["response"]
    texts = []

    for input, response in zip(inputs, responses):
        # EOS_TOKEN 추가
        text = alpaca_prompt.format(input=input, response=response) + EOS_TOKEN
        texts.append(text)

    return {"text": texts}

from datasets import load_dataset

dataset = load_dataset("junghwa28/recipes", split="train")
dataset = dataset.map(formatting_prompts_func, batched=True, remove_columns=dataset.column_names)
''')

st.image("images/modelpre.jpg")

st.markdown("* * *")
st.subheader("|모델 준비")
st.write("""이제 Huggingface TRL의 SFTTrainer를 사용해 봅시다!
        우리는 작업 속도를 높이기 위해 60 스텝만 실행하지만,
          전체 실행을 위해서는 num_train_epochs=1로 설정하고 max_steps=None으로 설정할 수 있습니다. 
         우리는 또한 TRL의 DPOTrainer도 지원합니다!)
""")
st.markdown("[자세한 내용]('https://huggingface.co/docs/trl/sft_trainer')")

st.code('''from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        # num_train_epochs = 1, # Set this for 1 full training run.
        max_steps = 60,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)''')


st.image("images/training.jpg")

st.markdown("* * *")
st.subheader("|모델 훈련")
st.code('''trainer_stats = trainer.train()''')

st.image("images/train.jpg")

st.markdown("* * *")

st.subheader("|alpaca_prompt 정의")
st.code('''
alpaca_prompt = """아래는 작업을 설명하는 지시사항입니다. 입력된 내용을 바탕으로 적절한 응답을 작성하세요.
### 지시사항:
{instruction}
### 입력:
{input}
### 응답:
"""

# FastLanguageModel 설정
FastLanguageModel.for_inference(model)  # Enable native 2x faster inference

# 입력 준비
instruction = "식단을 추천해주는 영양전문가입니다."
input_text = "고혈압 예방을 위한 저염식 레시피를 알려줘."  # 빈 문자열 대신 간단한 질문으로 대체

# 토큰화
inputs = tokenizer(
    [
        alpaca_prompt.format(
            instruction=instruction,
            input=input_text
        )
    ],
    return_tensors="pt"
).to("cuda")

# TextStreamer 설정
from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer)

# 생성
outputs = model.generate(
    **inputs,
    streamer=text_streamer,
    max_new_tokens=500,
    use_cache=True
)

# 생성된 텍스트 출력 (선택적)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\n완성된 텍스트:")
print(generated_text)''')

st.image("images/tuning.jpg")

st.markdown("* * *")
st.subheader("|gguf로 저장")

st.write('''GGUF / llama.cpp Conversion
To save to GGUF / llama.cpp, we support it natively now! We clone llama.
        cpp and we default save it to q8_0. We allow all methods like q4_k_m. 
        Use save_pretrained_gguf for local saving and push_to_hub_gguf for uploading to HF.)''')

st.code('''
        # Save to 8bit Q8_0
if False: model.save_pretrained_gguf("finalrecipes", tokenizer,)
# Remember to go to https://huggingface.co/settings/tokens for a token!
# And change hf to your username!
if False: model.push_to_hub_gguf("junghwa28/finalrecipes", tokenizer, token = "")

# Save to 16bit GGUF
if False: model.save_pretrained_gguf("model", tokenizer, quantization_method = "f16")
if False: model.push_to_hub_gguf("junghwa28/finalrecipes", tokenizer, quantization_method = "f16", token = "")

# Save to q4_k_m GGUF
if False: model.save_pretrained_gguf("model", tokenizer, quantization_method = "q4_k_m")
if False: model.push_to_hub_gguf("junghwa28/finalrecipes", tokenizer, quantization_method = "q4_k_m", token = "")

if True:
    model.push_to_hub_gguf(
        "junghwa28/finalrecipes", # Change hf to your username!
        tokenizer,
        quantization_method = "q8_0",
        token = "hf_ID", # Get a token at https://huggingface.co/settings/tokens
    ))''')

st.image("images/gguf.jpg")

st.markdown("* * *")
st.markdown()
