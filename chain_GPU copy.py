import os
os.environ["HF_HOME"] = "D:/Document/huggingface_cache"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Token để truy cập Hugging Face (giữ bí mật)
token = "token trên huggingface"

# Model path trên Hugging Face
# model_path = "mistralai/Mistral-7B-Instruct-v0.1"
model_path = "TheBloke/Mistral-7B-Instruct-v0.1-GPTQ"
# model_path = "vinai/PhoGPT-4B"
# model_path = "vnai/PhoGPT-7B5-Instruct"
# model_path = "vinai/PhoGPT-4B-Chat"

# https://huggingface.co/VietAI/envit5-translation 

# Tải tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Load mô hình với cấu hình 4-bit + GPU
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    device_map="auto"
)

print("Thiết bị đang sử dụng cho từng layer:")
print(model.hf_device_map)

# Tạo pipeline sinh văn bản
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
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
