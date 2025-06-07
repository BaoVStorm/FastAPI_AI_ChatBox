# # chạy 1 lần
# npm install -g cloudflared
# cloudflared login                             # để tạo 1 url cố định
# cloudflared tunnel create my-tunnel           # tạo 1 url cố định
# cloudflared tunnel run my-tunnel              # chạy tunnel

# # chạy nhiều lần
# chạy server
# cloudflared tunnel --url http://localhost:8000 | tee link.txt


from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import Optional
import uvicorn
import os

from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

os.environ["HF_HOME"] = "D:/Document/huggingface_cache"

app = FastAPI()

# Biến toàn cục
llm_chain = None

class ChatRequest(BaseModel):
    query: str

def load_llm(model_file):
    return CTransformers(
        model=model_file,
        model_type="llama",
        max_new_tokens=512,
        temperature=0.05
    )

def creat_prompt(template):
    return PromptTemplate(template=template, input_variables=["context", "question"])

def read_vectors_db():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local("vectorstores/db_chatBox", embedding_model, allow_dangerous_deserialization=True)

def create_qa_chain(prompt, llm, db):
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 1}, max_tokens_limit=1024),
        return_source_documents=False,
        chain_type_kwargs={'prompt': prompt}
    )

# ==== FastAPI startup event ====
@app.on_event("startup")
def startup_event():
    global llm_chain
    db = read_vectors_db()
    llm = load_llm("models/vinallama-7b-chat_q5_0.gguf")
    template = """<|im_start|>system\nSử dụng thông tin sau đây để trả lời câu hỏi. Nếu bạn không biết câu trả lời, hãy nói không biết, đừng cố tạo ra câu trả lời\n{context}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant"""
    prompt = creat_prompt(template)
    llm_chain = create_qa_chain(prompt, llm, db)

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        response = llm_chain.invoke({"query": request.query})
        return {"response": response}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run("server_cloudflare:app", host="127.0.0.1", port=8000, reload=True)
