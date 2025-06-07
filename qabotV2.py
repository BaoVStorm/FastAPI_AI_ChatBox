import os
os.environ["HF_HOME"] = "D:/Document/huggingface_cache"

from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Cấu hình
model_file = "models/vinallama-7b-chat_q5_0.gguf"
vector_db_path = "vectorstores/astrolingo_clean"

# Load LLM
def load_llm(model_file):
    llm = CTransformers(
        model=model_file,
        model_type="llama",
        max_new_tokens=512,
        temperature=0.05
    )
    return llm

# Tạo prompt template
def create_prompt(template):
    return PromptTemplate(template=template, input_variables=["context", "question"])

# Tạo RetrievalQA chain
def create_qa_chain(prompt, llm, db):
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(
            search_kwargs={"k": 3}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

# Load vector database
def read_vectors_db():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    db = FAISS.load_local(vector_db_path, embedding_model, allow_dangerous_deserialization=True)
    return db

# Chạy chính
if __name__ == "__main__":
    db = read_vectors_db()
    llm = load_llm(model_file)

    # Prompt cấu trúc chuẩn cho model chat
    template = """<|im_start|>system
Sử dụng thông tin sau đây để trả lời câu hỏi. Nếu bạn không biết câu trả lời, hãy nói bạn không biết.
{context}<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant"""
    prompt = create_prompt(template)

    llm_chain = create_qa_chain(prompt, llm, db)

    # Câu hỏi
    question = "Astrolingo là gì?"

    # Gọi chain
    response = llm_chain.invoke({"query": question})

    print("\n📌 Câu trả lời:")
    print(response["result"])

    print("\n📄 Nguồn:")
    for doc in response["source_documents"]:
        print(f"- Page {doc.metadata.get('page_label')}: {doc.page_content[:200]}...")