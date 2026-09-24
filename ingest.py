import shutil
import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from config import embeddings
import json

# 1. Load all PDFs
loader = DirectoryLoader(
    "documents/",
    glob="*.pdf",
    loader_cls=PyPDFLoader
)

docs = loader.load()

print(f"Loaded {len(docs)} pages")
print(docs[0].metadata)

# 2. Split documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

documents = text_splitter.split_documents(docs)

print(f"Created {len(documents)} chunks")
print(documents[0].metadata)

# Save chunks for BM25 keyword search
chunks_data = []

for doc in documents:
    chunks_data.append({
        "page_content": doc.page_content,
        "metadata": doc.metadata
    })

with open(
    "chunks.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        chunks_data,
        f,
        ensure_ascii=False,
        indent=2
    )

print("Chunks saved to chunks.json")

# 3. Delete old Chroma database
if os.path.exists("vectorstore"):
    shutil.rmtree("vectorstore")
    print("Old vector database deleted.")

# 4. Create Chroma database
db = Chroma.from_documents(
    documents,
    embeddings,
    persist_directory="vectorstore"
)

print("Vector database created successfully!")