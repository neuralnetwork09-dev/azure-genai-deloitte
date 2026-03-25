# =============================================================================
# PURPOSE
# This script upgrades the basic RAG pipeline from Day 2 into a smarter,
# self-correcting version. It adds three quality steps: first it generates
# a better search query using AI (HyDE), then it checks whether each retrieved
# document is actually relevant (grading), and finally it checks whether the
# generated answer is supported by the documents (hallucination detection).
# If retrieval quality is poor, the pipeline rewrites the query and tries again.
# =============================================================================

# os: reads your API keys and endpoints from the .env file
import os

# TypedDict, List: used to define the State — the shared notepad for all nodes
from typing import TypedDict, List

# load_dotenv: loads your .env file so os.getenv() can read your credentials
from dotenv import load_dotenv

# StateGraph: builds the workflow graph | END: marks where the workflow finishes
from langgraph.graph import StateGraph, END

# AzureChatOpenAI: connects to GPT-4o for generating text
# AzureOpenAIEmbeddings: converts text into vectors for similarity search
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

# SearchClient: connects to Azure AI Search to retrieve documents
from azure.search.documents import SearchClient

# VectorizedQuery: wraps a vector so Azure AI Search can do similarity search
from azure.search.documents.models import VectorizedQuery

# AzureKeyCredential: securely passes your API key to Azure services
from azure.core.credentials import AzureKeyCredential

# Load credentials from the .env file into memory
load_dotenv()

# Connect to GPT-4o — temperature=0 means consistent, factual answers every time
llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    temperature=0
)

# Connect to the embeddings model — converts text into a list of numbers (vector)
embeddings = AzureOpenAIEmbeddings(
    azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
)

# Connect to Azure AI Search — this is where your indexed documents live
search_client = SearchClient(
    endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
    index_name="documents-index",
    credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_KEY"))
)


# =============================================================================
# STATE — the shared notepad that every node in this workflow reads and writes
# Think of it as a whiteboard — each node picks up where the previous one left off
# =============================================================================
class AdvancedRAGState(TypedDict):
    query:          str        # the original question the user asked
    hyde_query:     str        # the AI-generated hypothetical answer used as search query
    retrieved_docs: List[dict] # all documents returned by Azure AI Search
    graded_docs:    List[dict] # only the documents that passed the relevance check
    response:       str        # the final answer generated for the user
    citations:      List[dict] # which documents were used to support the answer
    hallucination:  str        # verdict: "GROUNDED" means safe, "HALLUCINATED" means flagged
    retry_count:    int        # tracks how many times we have retried the search


# =============================================================================
# NODE 1 — HyDE: Generate a better search query
# The user's question is often too vague to search with directly.
# This node asks GPT-4o to write a hypothetical answer first.
# That hypothetical answer uses the same vocabulary as the real documents,
# so it retrieves much more relevant results when used as the search query.
# =============================================================================
def generate_hyde_query(state: AdvancedRAGState) -> dict:
    print(f"[HyDE] Generating hypothetical document for query: {state['query']}")

    # Ask GPT-4o to write a short answer as if it were a document in the system
    prompt = f"""Write a short 2-3 sentence hypothetical answer to the following
question as if you were a document in an enterprise knowledge base.
Use formal language and domain terminology.

Question: {state["query"]}

Hypothetical answer:"""

    result = llm.invoke(prompt)
    hyde_query = result.content.strip()
    print(f"[HyDE] Hypothetical query: {hyde_query[:100]}...")

    # Save the hypothetical answer as the query for the next retrieval step
    return {"hyde_query": hyde_query, "retry_count": state.get("retry_count", 0)}


# =============================================================================
# NODE 2 — Retrieve: Search Azure AI Search for relevant documents
# Uses the HyDE query (or rewritten query on retry) to run a hybrid search.
# Hybrid search combines keyword matching AND vector similarity for best results.
# =============================================================================
def retrieve_documents(state: AdvancedRAGState) -> dict:
    # Use the HyDE query on the first attempt, or the rewritten query on retry
    search_query = state.get("hyde_query") or state["query"]
    print(f"[Retrieve] Searching with: {search_query[:80]}...")

    # Convert the search query into a vector (list of numbers)
    vector = embeddings.embed_query(search_query)

    # Wrap the vector so Azure AI Search knows how to use it
    vec_query = VectorizedQuery(vector=vector, k_nearest_neighbors=5, fields="content_vector")

    # Run hybrid search — keyword and vector at the same time
    results = search_client.search(
        search_text=search_query,
        vector_queries=[vec_query],
        top=5,
        select=["id","content","document_type","source_file","chunk_index"]
    )

    # Convert search results into a clean list of dictionaries
    docs = [{"id":r["id"],"content":r["content"],"source_file":r["source_file"],
             "document_type":r["document_type"],"chunk_index":r["chunk_index"],
             "search_score":r["@search.score"]} for r in results]

    print(f"[Retrieve] Found {len(docs)} documents")
    return {"retrieved_docs": docs}


# =============================================================================
# NODE 3 — Grade: Check each document for relevance
# This is the quality gate. Not every retrieved document is useful.
# Each document is sent to GPT-4o with the original query and a simple question:
# is this document relevant? Only documents that pass go forward.
# =============================================================================
def grade_documents(state: AdvancedRAGState) -> dict:
    print(f"[Grade] Grading {len(state['retrieved_docs'])} documents...")
    graded = []

    for doc in state["retrieved_docs"]:
        # Ask GPT-4o: does this document help answer the user's question?
        prompt = f"""You are a relevance grader for a document retrieval system.

Query: {state["query"]}

Document excerpt:
{doc["content"][:400]}

Is this document relevant to the query? Reply with only: YES or NO"""

        result = llm.invoke(prompt)

        # Keep the document only if GPT-4o said YES
        if "YES" in result.content.upper():
            doc["relevant"] = True
            graded.append(doc)
            print(f"  PASS: {doc['source_file']} chunk {doc['chunk_index']}")
        else:
            print(f"  FAIL: {doc['source_file']} chunk {doc['chunk_index']} — not relevant")

    print(f"[Grade] {len(graded)} documents passed relevance check")
    return {"graded_docs": graded}


# =============================================================================
# NODE 4 — Rewrite: Improve the query and try again
# Called only when grading found too few relevant documents.
# GPT-4o rewrites the original query to be more specific and domain-aware.
# The rewritten query is stored back into hyde_query so Node 2 uses it next.
# =============================================================================
def rewrite_query(state: AdvancedRAGState) -> dict:
    retry = state.get("retry_count", 0) + 1
    print(f"[Rewrite] Retry {retry} — rewriting query for better retrieval")

    # Ask GPT-4o to make the question more specific and searchable
    prompt = f"""Rewrite the following query to improve document retrieval.
Make it more specific. Add domain terminology. Include key entity names.
Return only the rewritten query.

Original: {state["query"]}

Rewritten:"""

    result = llm.invoke(prompt)
    rewritten = result.content.strip()
    print(f"[Rewrite] New query: {rewritten}")

    # Store the rewritten query — Node 2 will use this on the next retrieval attempt
    return {"hyde_query": rewritten, "retry_count": retry}


# =============================================================================
# NODE 5 — Generate: Write the final answer using only the graded documents
# This node uses only documents that passed the relevance check.
# The prompt tells GPT-4o to cite every claim with [SOURCE N] — no making things up.
# =============================================================================
def generate_answer(state: AdvancedRAGState) -> dict:
    # Use graded documents if available, otherwise fall back to all retrieved docs
    docs = state.get("graded_docs") or state.get("retrieved_docs", [])
    print(f"[Generate] Generating answer from {len(docs)} graded documents")

    if not docs:
        return {"response": "I could not find relevant documents to answer this question.",
                "citations": []}

    # Format each document with a source label so GPT-4o can reference it
    context = "\n\n".join([f"[SOURCE {i+1}: {d['source_file']}]\n{d['content']}"
                            for i, d in enumerate(docs)])

    # The prompt enforces grounding — GPT-4o must only use what is in the sources
    prompt = f"""You are an intelligent document assistant for a financial services
and healthcare organisation. Answer the question using ONLY the sources below.
Cite every factual claim with [SOURCE N]. If the answer is not in the sources, say so.

Question: {state["query"]}

Sources:
{context}

Answer:"""

    result = llm.invoke(prompt)

    # Find which source numbers appear in the response and build the citations list
    citations = [{"source_number":i+1,"source_file":d["source_file"],
                  "excerpt":d["content"][:150]}
                 for i,d in enumerate(docs)
                 if f"[SOURCE {i+1}]" in result.content]

    return {"response": result.content, "citations": citations}


# =============================================================================
# NODE 6 — Hallucination Detector: Did the answer stay within the sources?
# A separate GPT-4o call reads the generated answer and the source documents
# and checks whether every claim in the answer is backed by the sources.
# Result is stored in state as part of the audit trail.
# =============================================================================
def detect_hallucination(state: AdvancedRAGState) -> dict:
    docs = state.get("graded_docs") or state.get("retrieved_docs", [])
    print("[Hallucination] Checking answer against sources...")

    if not docs:
        return {"hallucination": "NO_SOURCES"}

    # Combine the first 300 characters of each source into one block for checking
    sources_text = "\n".join([d["content"][:300] for d in docs])

    # Ask GPT-4o to compare the answer against the sources
    prompt = f"""You are a hallucination detector for an AI system.

Answer: {state["response"]}

Sources:
{sources_text}

Does the answer contain only information present in the sources above?
Reply with only: GROUNDED or HALLUCINATED"""

    result = llm.invoke(prompt)

    # GROUNDED = answer is supported by sources | HALLUCINATED = answer invented something
    verdict = "GROUNDED" if "GROUNDED" in result.content.upper() else "HALLUCINATED"
    print(f"[Hallucination] Verdict: {verdict}")

    return {"hallucination": verdict}


# =============================================================================
# ROUTING FUNCTION — decides which node runs after document grading
# Reads the graded_docs count and retry_count from state.
# Returns a string that exactly matches one of the node names above.
# =============================================================================
def route_after_grading(state: AdvancedRAGState) -> str:
    graded = state.get("graded_docs", [])
    retry = state.get("retry_count", 0)

    # Enough relevant documents found — move on to generating the answer
    if len(graded) >= 2:
        return "generate_answer"

    # Tried twice already — generate with whatever we have rather than looping forever
    if retry >= 2:
        print("[Route] Max retries reached — generating with available documents")
        return "generate_answer"

    # Not enough docs and retries remaining — rewrite the query and try again
    return "rewrite_query"


# =============================================================================
# BUILD THE GRAPH — wire all six nodes together
# =============================================================================
graph = StateGraph(AdvancedRAGState)

# Register each node by name — the name is what routing functions return
graph.add_node("generate_hyde_query",  generate_hyde_query)
graph.add_node("retrieve_documents",   retrieve_documents)
graph.add_node("grade_documents",      grade_documents)
graph.add_node("rewrite_query",        rewrite_query)
graph.add_node("generate_answer",      generate_answer)
graph.add_node("detect_hallucination", detect_hallucination)

# The workflow always starts at the HyDE query generator
graph.set_entry_point("generate_hyde_query")

# Fixed path: HyDE -> retrieve -> grade
graph.add_edge("generate_hyde_query", "retrieve_documents")
graph.add_edge("retrieve_documents",  "grade_documents")

# Decision point: after grading, either rewrite+retry or generate
graph.add_conditional_edges("grade_documents", route_after_grading)

# If rewriting, loop back to retrieval for another attempt
graph.add_edge("rewrite_query",        "retrieve_documents")

# After generating the answer, always run hallucination detection
graph.add_edge("generate_answer",      "detect_hallucination")

# Hallucination detection is the last step — workflow ends here
graph.add_edge("detect_hallucination", END)

# Lock the graph into a runnable application
advanced_rag = graph.compile()


# =============================================================================
# RUN THE PIPELINE — test with three different queries
# =============================================================================
if __name__ == "__main__":
    test_queries = [
        "What is the total amount due on the Contoso Consulting invoice?",
        "What medications is the patient Sarah Johnson currently taking?",
        "What are the payment terms for the invoice from Contoso?",
    ]

    print("=== Advanced Self-Correcting RAG Pipeline ===")

    for query in test_queries:
        print("\n" + "="*60)
        print(f"Query: {query}")
        print("="*60)

        # Run the full graph — pass initial empty values for all state fields
        result = advanced_rag.invoke({
            "query":          query,
            "hyde_query":     "",
            "retrieved_docs": [],
            "graded_docs":    [],
            "response":       "",
            "citations":      [],
            "hallucination":  "",
            "retry_count":    0,
        })

        # Print the final answer and quality verdicts
        print(f"\nAnswer:\n{result['response']}")
        print(f"\nHallucination check: {result['hallucination']}")

        # Show which source documents were cited in the answer
        if result["citations"]:
            print("Citations:")
            for c in result["citations"]:
                print(f"  [SOURCE {c['source_number']}] {c['source_file']}")
