import os
os.environ["HF_HOME"] = "D:/Document/huggingface_cache"


# https://huggingface.co/vinai/phobert-base

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig, AutoConfig
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

model_path = "vinai/PhoGPT-4B-Chat"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
# Chỉ load model thôi, không cần load thêm config
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    trust_remote_code=False,
    quantization_config=bnb_config
)

# Kiểm tra từ model.config
is_seq2seq = getattr(model.config, "is_encoder_decoder", False)
print("is_seq2seq:", is_seq2seq)   # False nghĩa là CausalLM, True nghĩa Seq2Seq

# Tạo pipeline dựa trên kết quả
task = "text-generation" if not is_seq2seq else "text2text-generation"
pipe = pipeline(
    task,
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    temperature=0.7,
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
