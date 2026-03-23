import os
import json
from typing import TypedDict, List
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential
 
load_dotenv()
 
# initialise Azure OpenAI LLM
llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    temperature=0
)
 
# initialise embeddings model
embeddings_model = AzureOpenAIEmbeddings(
    azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
)
 
# initialise Azure Search client
search_client = SearchClient(
    endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
    index_name="documents-index",
    credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_KEY"))
)
 
 
# define pipeline state
class RAGState(TypedDict):
    query: str
    rewritten_query: str
    retrieved_docs: List[dict]
    response: str
    citations: List[dict]
    retry_count: int
    sufficient: bool
 
 
# rewrite query and increment retry count
def rewrite_query(state: RAGState) -> dict:
    print(f"[rewrite_query] Original: {state['query']}")
 
    prompt = f"""Rewrite the query to improve search results.
Make it more specific and use better keywords.

Original query: {state["query"]}

Rewritten query:"""
 
    result = llm.invoke(prompt)
    rewritten = result.content.strip()
 
    print(f"[rewrite_query] Rewritten: {rewritten}")
 
    return {
        "rewritten_query": rewritten,
        "retry_count": state.get("retry_count", 0) + 1
    }
 
 
# retrieve documents using hybrid search
def retrieve_documents(state: RAGState) -> dict:
    query = state.get("rewritten_query") or state["query"]
    print(f"[retrieve_documents] Searching for: {query}")
 
    vector = embeddings_model.embed_query(query)
 
    vec_query = VectorizedQuery(
        vector=vector,
        k_nearest_neighbors=5,
        fields="content_vector"
    )
 
    results = search_client.search(
        search_text=query,
        vector_queries=[vec_query],
        top=5,
        select=["id", "content", "document_type", "source_file", "chunk_index"]
    )
 
    docs = []
    for r in results:
        docs.append({
            "id": r["id"],
            "content": r["content"],
            "document_type": r["document_type"],
            "source_file": r["source_file"],
            "chunk_index": r["chunk_index"],
            "search_score": r["@search.score"],
        })
 
    print(f"[retrieve_documents] Retrieved {len(docs)} documents")
    return {"retrieved_docs": docs}
 
 
# grade if documents are useful
def grade_documents(state: RAGState) -> dict:
    docs = state.get("retrieved_docs", [])
    retry_count = state.get("retry_count", 0)
 
    if not docs:
        print("[grade_documents] No documents found")
        return {"sufficient": False, "retry_count": retry_count}
 
    context_preview = "\n".join([d["content"][:200] for d in docs[:3]])
 
    prompt = f"""Check if these documents are even partially useful.

Query: {state["query"]}

Documents:
{context_preview}

Reply YES or NO"""
 
    result = llm.invoke(prompt)
    is_sufficient = "YES" in result.content.upper()
 
    print(f"[grade_documents] Sufficient: {is_sufficient} Retry: {retry_count}")
 
    return {
        "sufficient": is_sufficient,
        "retry_count": retry_count
    }
 
 
# generate answer
def generate_response(state: RAGState) -> dict:
    docs = state.get("retrieved_docs", [])
    print(f"[generate_response] Using {len(docs)} documents")
 
    if not docs:
        return {
            "response": "No relevant documents found.",
            "citations": []
        }
 
    context_parts = []
    for i, doc in enumerate(docs):
        context_parts.append(f"[SOURCE {i+1}] {doc['content']}")
 
    context = "\n\n".join(context_parts)
 
    prompt = f"""Answer using only the sources below.

Question: {state["query"]}

Sources:
{context}

Answer:"""
 
    result = llm.invoke(prompt)
 
    return {"response": result.content}
 
 
# extract citations
def extract_citations(state: RAGState) -> dict:
    docs = state.get("retrieved_docs", [])
    citations = []
 
    for i, doc in enumerate(docs, 1):
        if f"[SOURCE {i}]" in state.get("response", ""):
            citations.append({
                "source_number": i,
                "source_file": doc["source_file"],
                "chunk_index": doc["chunk_index"],
                "excerpt": doc["content"][:200],
            })
 
    print(f"[extract_citations] Found {len(citations)} citations")
 
    return {"citations": citations}
 
 
# routing logic to stop infinite loop
def route_after_grading(state: RAGState) -> str:
    if state.get("sufficient", False):
        return "generate_response"
 
    if state.get("retry_count", 0) >= 2:
        print("[route] Max retries reached, proceeding anyway")
        return "generate_response"
 
    return "rewrite_query"
 
 
# build graph
graph = StateGraph(RAGState)
 
graph.add_node("rewrite_query", rewrite_query)
graph.add_node("retrieve_documents", retrieve_documents)
graph.add_node("grade_documents", grade_documents)
graph.add_node("generate_response", generate_response)
graph.add_node("extract_citations", extract_citations)
 
graph.set_entry_point("rewrite_query")
graph.add_edge("rewrite_query", "retrieve_documents")
graph.add_edge("retrieve_documents", "grade_documents")
graph.add_conditional_edges("grade_documents", route_after_grading)
graph.add_edge("generate_response", "extract_citations")
graph.add_edge("extract_citations", END)
 
rag_app = graph.compile()
 
 
# run pipeline
if __name__ == "__main__":
    queries = [
        "What is the total amount due on the Contoso invoice?",
        "What medications is the patient currently taking?",
        "What are the payment terms and bank details?"
    ]
 
    print("=== RAG Pipeline Fixed ===")
 
    for query in queries:
        print("\n" + "="*50)
        print(f"Query: {query}")
        print("="*50)
 
        result = rag_app.invoke({
            "query": query,
            "rewritten_query": "",
            "retrieved_docs": [],
            "response": "",
            "citations": [],
            "retry_count": 0,
            "sufficient": False,
        })
 
        print("\nResponse:\n", result["response"])
 
        if result["citations"]:
            print("\nCitations:")
            for c in result["citations"]:
                print(f"[SOURCE {c['source_number']}] {c['source_file']} chunk {c['chunk_index']}")
                print(f"Excerpt: {c['excerpt'][:100]}...")