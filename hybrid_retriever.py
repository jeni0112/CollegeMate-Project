import json
import re
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from langchain_community.vectorstores import Chroma

from config import embeddings


# -----------------------------------------
# 1. Load Chroma
# -----------------------------------------

db = Chroma(
    persist_directory="vectorstore",
    embedding_function=embeddings
)


# -----------------------------------------
# 2. Load chunks for BM25
# -----------------------------------------

with open(
    "chunks.json",
    "r",
    encoding="utf-8"
) as file:

    chunks = json.load(file)


# -----------------------------------------
# 3. Tokenizer
# -----------------------------------------

def tokenize(text):

    text = text.lower()

    text = re.sub(
        r"[^\w\s]",
        " ",
        text
    )

    return text.split()


# -----------------------------------------
# 4. Prepare BM25 documents
# -----------------------------------------

documents = [
    chunk["page_content"]
    for chunk in chunks
]


tokenized_documents = [
    tokenize(document)
    for document in documents
]


# -----------------------------------------
# 5. Create BM25 index
# -----------------------------------------

bm25 = BM25Okapi(
    tokenized_documents
)


# -----------------------------------------
# 6. Hybrid search
# -----------------------------------------

def hybrid_search(
    query,
    k=5
):

    # -------------------------------
    # Semantic search
    # -------------------------------

    semantic_results = db.similarity_search(
        query,
        k=k
    )


    # -------------------------------
    # BM25 search
    # -------------------------------

    tokenized_query = tokenize(query)

    bm25_results = bm25.get_top_n(
        tokenized_query,
        documents,
        n=k
    )


    # -------------------------------
    # Combine results using RRF
    # -------------------------------

    scores = {}

    documents_map = {}


    # Semantic results

    for rank, doc in enumerate(
        semantic_results,
        start=1
    ):

        content = doc.page_content

        documents_map[content] = doc

        scores[content] = scores.get(
            content,
            0
        ) + 1 / (60 + rank)


    # BM25 results

    for rank, content in enumerate(
        bm25_results,
        start=1
    ):

        scores[content] = scores.get(
            content,
            0
        ) + 1 / (60 + rank)


        if content not in documents_map:

            for chunk in chunks:

                 if chunk["page_content"] == content:
                     documents_map[content] = Document(
                         page_content=chunk["page_content"],
                         metadata=chunk["metadata"]
                        
                    )
                     break


    # -------------------------------
    # Sort by RRF score
    # -------------------------------

    ranked_documents = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )


    # -------------------------------
    # Return top k
    # -------------------------------

    final_results = []

    for content, score in ranked_documents[:k]:

        doc = documents_map[content]

        final_results.append(
            (
                doc,
                score
            )
        )


    return final_results