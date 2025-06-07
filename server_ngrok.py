
# 
# pip install pyngrok
# ngrok config add-authtoken <your-token>

from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import Optional
import uvicorn
from pyngrok import ngrok

from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

import os
os.environ["HF_HOME"] = "D:/Document/huggingface_cache"

# ==== Config ====
model_file = "models/vinallama-7b-chat_q5_0.gguf"
vector_db_path = "vectorstores/db_chatBox"

# ==== Khởi tạo LLM ====
def load_llm(model_file):
    return CTransformers(
        model=model_file,
        model_type="llama",
        max_new_tokens=512,
        temperature=0.05
    )

# ==== Prompt Template ====
def creat_prompt(template):
    return PromptTemplate(template=template, input_variables=["context", "question"])

# ==== Load Vector DB ====
def read_vectors_db():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local(vector_db_path, embedding_model, allow_dangerous_deserialization=True)

# ==== Tạo QA Chain ====
def create_qa_chain(prompt, llm, db):
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 1}, max_tokens_limit=1024),
        return_source_documents=False,
        chain_type_kwargs={'prompt': prompt}
    )

# ==== Khởi tạo FastAPI ====
app = FastAPI()

# Khởi tạo model, db, prompt, chain
db = read_vectors_db()
llm = load_llm(model_file)
template = """<|im_start|>system\nSử dụng thông tin sau đây để trả lời câu hỏi. Nếu bạn không biết câu trả lời, hãy nói không biết, đừng cố tạo ra câu trả lời\n{context}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant"""
prompt = creat_prompt(template)
llm_chain = create_qa_chain(prompt, llm, db)

# ==== Schema input ====
class ChatRequest(BaseModel):
    query: str

# ==== Endpoint /chat ====
@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        response = llm_chain.invoke({"query": request.query})
        return {"response": response}
    except Exception as e:
        return {"error": str(e)}

# ==== Chạy local ====
if __name__ == "__main__":
    public_url = ngrok.connect(8000)
    print(f"🚀 Ngrok public URL: {public_url}")

    uvicorn.run("server_ngrok:app", host="127.0.0.1", port=8000, reload=True)
