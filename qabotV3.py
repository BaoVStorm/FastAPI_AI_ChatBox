import os
os.environ["HF_HOME"] = "D:/Document/huggingface_cache"

from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

# Cấu hình
model_file = "models/vinallama-7b-chat_q5_0.gguf"
vector_db_path = "vectorstores/astrolingo_clean"

# Load LLM
def load_llm():
    return CTransformers(
        model=model_file,
        model_type="llama",
        max_new_tokens=512,
        temperature=0.05
    )

# Prompt template rõ ràng, không dùng <|im_start|>
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

# Load vector DB
def read_vector_db():
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    return FAISS.load_local(vector_db_path, embedding, allow_dangerous_deserialization=True)

# Tạo QA chain
def create_qa_chain(llm, prompt, db):
    retriever = db.as_retriever(search_kwargs={"k": 1})
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

# Main
if __name__ == "__main__":
    db = read_vector_db()
    llm = load_llm()
    prompt = create_prompt()
    qa_chain = create_qa_chain(llm, prompt, db)

    # Đặt câu hỏi
    question = "AstroLingo là gì?"
    response = qa_chain.invoke({"query": question})

    print("\n📌 Câu trả lời:")
    print(response["result"])

    print("\n📄 Trích từ:")
    for doc in response["source_documents"]:
        print(f"- Trang: {doc.metadata.get('page', '?')}")
        print(f"  >> {doc.page_content.strip()[:300]}...\n")
