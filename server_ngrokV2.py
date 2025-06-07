import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from pyngrok import ngrok
import uvicorn

# ==== Cấu hình đường dẫn ====
os.environ["HF_HOME"] = "D:/Document/huggingface_cache"
model_file = "models/vinallama-7b-chat_q5_0.gguf"
vector_db_path = "vectorstores/astrolingo_clean"

# ==== Load LLM ====
def load_llm():
    return CTransformers(
        model=model_file,
        model_type="llama",
        max_new_tokens=512,
        temperature=0.05
    )

# ==== Prompt rõ ràng ====
def create_prompt():
    template = """
Bạn là một trợ lý AI hữu ích. Dựa trên thông tin sau, hãy trả lời câu hỏi ngắn gọn và chính xác.
Nếu không có đủ thông tin, hãy nói "Tôi không biết".

Thông tin:
{context}

Câu hỏi:
{question}

Trả lời:
"""
    return PromptTemplate(template=template, input_variables=["context", "question"])

# ==== Load FAISS vector DB ====
def read_vector_db():
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    return FAISS.load_local(vector_db_path, embedding, allow_dangerous_deserialization=True)

# ==== Tạo QA chain ====
def create_qa_chain(llm, prompt, db):
    retriever = db.as_retriever(search_kwargs={"k": 1})
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

# ==== Khởi tạo FastAPI ====
app = FastAPI()

# Load các thành phần sẵn khi app khởi chạy
db = read_vector_db()
llm = load_llm()
prompt = create_prompt()
qa_chain = create_qa_chain(llm, prompt, db)

# ==== Định nghĩa request model ====
class ChatRequest(BaseModel):
    query: str

# ==== API endpoint ====
@app.post("/chat")
async def chat(req: ChatRequest):
    try:
        result = qa_chain.invoke({"query": req.query})
        response_text = result["result"]
        sources = [
            {
                "page": doc.metadata.get("page", "?"),
                "content": doc.page_content.strip()[:300]
            }
            for doc in result["source_documents"]
        ]
        return {"response": response_text, "sources": sources}
    except Exception as e:
        return {"error": str(e)}

# ==== Chạy local cùng ngrok ====
if __name__ == "__main__":
    public_url = ngrok.connect(8000)
    print(f"🔗 Ngrok public URL: {public_url}")
    uvicorn.run(app, host="127.0.0.1", port=8000)
