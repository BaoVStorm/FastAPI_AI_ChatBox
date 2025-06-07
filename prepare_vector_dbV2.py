import os
os.environ["HF_HOME"] = "D:/Document/huggingface_cache"

from langchain_community.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

# Cấu hình đường dẫn
docx_path = "data/AstroLingo.docx"
vector_db_path = "vectorstores/astrolingo_clean"

def create_db_from_docx():
    print("📄 Đang tải file DOCX...")
    loader = Docx2txtLoader(docx_path)
    documents = loader.load()

    print("✂️ Đang chia nhỏ văn bản...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    print("🧠 Đang nhúng embedding...")
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    db = FAISS.from_documents(chunks, embedding)

    print("💾 Đang lưu vector database...")
    db.save_local(vector_db_path)
    print("✅ Vector DB đã sẵn sàng!")

if __name__ == "__main__":
    create_db_from_docx()
