import os
os.environ["HF_HOME"] = "D:/Document/huggingface_cache"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig, AutoConfig
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Token để truy cập Hugging Face (giữ bí mật)
token = "token trên huggingface"

# Model path trên Hugging Face
# model_path = "mistralai/Mistral-7B-Instruct-v0.1"
# model_path = "vinai/PhoGPT-4B"
# model_path = "vnai/PhoGPT-7B5-Instruct"
model_path = "vinai/PhoGPT-4B-Chat"

# model_path = "eunyounglee/GPT-NeoX-1.3B-Viet-1"


# ----- chấm bài thi

# model_path = "dangvantuan/vietnamese-document-embedding"
# model_path = "vinai/phobert-base"

#  ------- dịch từ

# model_path = "VietAI/envit5-translation"
# model_path = "vinai/vinai-translate-vi2en"
# model_path = "Helsinki-NLP/opus-mt-en-vi"


config = AutoConfig.from_pretrained(model_path, token=token, trust_remote_code=True)
model_type = config.architectures[0]
print("Loại mô hình:", model_type)

# if "CausalLM" in model_type:
#     model = AutoModelForCausalLM.from_pretrained(...)
# elif "Seq2Seq" in model_type:
#     model = AutoModelForSeq2SeqLM.from_pretrained(...)
# else:
#     print("Mô hình không hỗ trợ sinh văn bản.")


# https://huggingface.co/VietAI/envit5-translation 

# Tải tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, token=token, trust_remote_code=True)

# BitsAndBytes cấu hình cho 4-bit
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Load mô hình với cấu hình 4-bit + GPU
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    trust_remote_code=True,
    token=token,
    quantization_config=bnb_config
)

# Tạo pipeline sinh văn bản
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=1024,
    temperature=0.1,
    do_sample=True,
    return_full_text=False
)

# Tích hợp LangChain
llm = HuggingFacePipeline(pipeline=pipe)

# Prompt template instruction-style
template = """<|im_start|>system
Bạn là một trợ lý AI hữu ích, chính xác và lịch sự.
<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant"""

prompt = PromptTemplate(template=template, input_variables=["question"])
chain = LLMChain(prompt=prompt, llm=llm)

# Test
question = "Thủ đô của Việt Nam là gì?"
response = chain.invoke({"question": question})
print(response["text"])
