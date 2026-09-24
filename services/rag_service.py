from langchain_community.vectorstores import Chroma
import json
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from config import embeddings
from services.openai_service import get_response

# Load the existing Chroma database
db = Chroma(
    persist_directory="vectorstore",
    embedding_function=embeddings
)

# Load chunks saved during ingestion
with open("chunks.json", "r", encoding="utf-8") as f:
    chunks_data = json.load(f)

documents = [
    Document(
        page_content=item["page_content"],
        metadata=item["metadata"]
    )
    for item in chunks_data
]

# Create BM25 keyword retriever
bm25_retriever = BM25Retriever.from_documents(
    documents
)

bm25_retriever.k = 5

def reciprocal_rank_fusion(vector_results,keyword_results,k=60):

    """
    Combine vector-search and BM25 results using
    Reciprocal Rank Fusion (RRF).
    """

    scores = {}
    documents_map = {}

    # Vector search ranking
    for rank, doc in enumerate(vector_results, start=1):

        doc_id = (
            doc.metadata.get("source", ""),
            doc.metadata.get("page", ""),
            doc.page_content
        )

        scores[doc_id] = scores.get(doc_id, 0) + (
            1 / (k + rank)
        )

        documents_map[doc_id] = doc


    # BM25 keyword ranking
    for rank, doc in enumerate(keyword_results, start=1):

        doc_id = (
            doc.metadata.get("source", ""),
            doc.metadata.get("page", ""),
            doc.page_content
        )

        scores[doc_id] = scores.get(doc_id, 0) + (
            1 / (k + rank)
        )

        documents_map[doc_id] = doc


    # Sort by combined RRF score
    ranked_documents = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return [
        documents_map[doc_id]
        for doc_id, score in ranked_documents
    ]

def rewrite_query(query, conversation_history):

    """
    Convert the current question into a standalone question
    using previous conversation context.
    """

    # If this is the first question, no rewriting is needed
    if not conversation_history:
        return query

    # Keep only recent conversation
    recent_history = conversation_history[-6:]

    history_text = ""

    for message in recent_history:

        role = message.get("role")
        content = message.get("content")

        history_text += f"{role}: {content}\n"

    rewrite_prompt = f"""
    You are a query rewriting assistant for a College Assistant chatbot.

    Your task is to rewrite the student's latest question into a
    standalone question that can be searched in a college knowledge base.

    Use the conversation history to understand references such as:

    - it
    - they
    - them
    - this
    - that
    - those
    - what about
    - how about

    Rules:

    1. Do NOT answer the question.
    2. Do NOT add information that is not present in the conversation.
    3. Preserve the student's original intent.
    4. If the question is already standalone, return it unchanged.
    5. Return ONLY the rewritten question.

    Conversation History:
    {history_text}

    Latest Student Question:
    {query}

    Standalone Question:
    """

    rewritten_query = get_response(rewrite_prompt)

    return rewritten_query.strip()


def get_rag_response(query,conversation_history):

    """
    Conversational RAG pipeline.

    1. Rewrite question using conversation history
    2. Retrieve relevant documents
    3. Filter by similarity threshold
    4. Build grounded context
    5. Generate answer
    6. Return answer + sources
    """

    # --------------------------------------------------
    # STEP 1: Rewrite query
    # --------------------------------------------------

    standalone_query = rewrite_query(
        query,
        conversation_history
    )

    print("\n==============================")
    print("Original Question:")
    print(query)

    print("\nStandalone Question:")
    print(standalone_query)
    print("==============================")    

    # --------------------------------------------------
    # STEP 2: Hybrid Retrieval
    # --------------------------------------------------

    # Vector search
    vector_results = db.similarity_search(standalone_query,k=5)

    # BM25 keyword search
    keyword_results = bm25_retriever.invoke(standalone_query)

    print("\n==============================")
    print("VECTOR SEARCH RESULTS")
    print("==============================")

    for i, doc in enumerate(vector_results, start=1):
        print(f"\n--- Vector Chunk {i} ---")    
        print("Metadata:", doc.metadata)
        print(doc.page_content[:500])


    print("\n==============================")
    print("BM25 KEYWORD RESULTS")
    print("==============================")

    for i, doc in enumerate(keyword_results, start=1):
        print(f"\n--- BM25 Chunk {i} ---")
        print("Metadata:", doc.metadata)
        print(doc.page_content[:500])


    # Combine using RRF
    retrieved_results = reciprocal_rank_fusion(vector_results,keyword_results)

    # Keep only top results
    retrieved_results = retrieved_results[:5]


    print("\n==============================")
    print("FINAL HYBRID RESULTS")
    print("==============================")

    for i, doc in enumerate(retrieved_results, start=1):
        print(f"\n--- Hybrid Chunk {i} ---")
        print("Metadata:", doc.metadata)
        print(doc.page_content[:500])

    # --------------------------------------------------
    # STEP 3: No relevant information found
    # --------------------------------------------------

    if not retrieved_results:

        return (
            "I couldn't find this information in the college documents.",
            []
        )


# --------------------------------------------------
# STEP 4: Build context
# --------------------------------------------------

    context_parts = []

    for result in retrieved_results:

        page = result.metadata.get(
            "page_label",
            result.metadata.get("page", "Unknown")
        )

        source = result.metadata.get(
            "source",
            "Unknown"
        )

        context_parts.append(
            f"Source: {source}\n"
            f"Page: {page}\n"
            f"Content:\n{result.page_content}"
        )


    context = "\n\n".join(context_parts)


    # --------------------------------------------------
    # STEP 5: Generate grounded answer
    # --------------------------------------------------

    answer_prompt = f"""
    You are a College Assistant.

    Answer the user's question using ONLY the information provided
    in the context below.

    IMPORTANT RULES:

    1. If the answer is clearly available in the context, answer the question directly.
    2. Do NOT say that the information could not be found if the context contains the answer.
    3. If the answer is NOT available in the context, respond ONLY with:
    "I couldn't find this information in the college documents."
    4. Do not add information that is not present in the context.
    5. Do not provide unrelated information.

    Format the answer for easy reading:

    - Use short paragraphs.
    - Use bullet points when listing multiple items.
    - Use headings when appropriate.
    - Leave a blank line between sections.

    College Information:
    {context}

    Student Question:
    {query}

    Answer:
    """

    response = get_response(answer_prompt)


    # --------------------------------------------------
    # STEP 6: Collect unique sources
    # --------------------------------------------------

    sources = []

    seen = set()

    for result in retrieved_results:

        page = result.metadata.get(
            "page_label",
            result.metadata.get("page", "Unknown")
        )

        source = result.metadata.get(
            "source",
            "Unknown"
        )

        key = (source, page)

        if key not in seen:

            seen.add(key)

            sources.append({
                "source": source,
                "page": page
            })


    # --------------------------------------------------
    # STEP 7: Return answer + sources
    # --------------------------------------------------

    return response, sources

    