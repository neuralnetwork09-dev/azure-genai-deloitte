# =============================================================================
# PURPOSE
# This script builds a conversational document assistant that remembers
# everything said in the conversation. When a user asks a follow-up like
# "What is the total on it?" — the assistant knows "it" means the Contoso
# invoice from the previous question. It retrieves documents from Azure AI
# Search for each question and generates grounded, context-aware answers.
# This becomes the conversational interface agent in the Day 4 multi-agent system.
# =============================================================================

# os: reads your API keys and endpoints from the .env file
import os

# TypedDict: defines the shape of the State (shared notepad)
# List: declares a field that holds a list of items
# Annotated: adds special behaviour to a field — used here to make messages accumulate
from typing import TypedDict, List, Annotated

# operator.add: used with Annotated to say "append new items, do not replace the list"
import operator

# load_dotenv: reads the .env file so credentials are available via os.getenv()
from dotenv import load_dotenv

# StateGraph: builds the workflow | END: marks where the workflow finishes
from langgraph.graph import StateGraph, END

# MemorySaver: saves the full conversation state after every turn
# so the same conversation can be resumed using the same session ID
from langgraph.checkpoint.memory import MemorySaver

# AzureChatOpenAI: connects to GPT-4o for generating answers
# AzureOpenAIEmbeddings: converts text into vectors for similarity search
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

# SearchClient: used to query the Azure AI Search index
from azure.search.documents import SearchClient

# VectorizedQuery: wraps a vector so Azure AI Search can find similar documents
from azure.search.documents.models import VectorizedQuery

# AzureKeyCredential: securely passes your API key to Azure services
from azure.core.credentials import AzureKeyCredential

# Load all credentials from the .env file into memory
load_dotenv()

# Connect to GPT-4o — temperature=0 gives consistent, factual responses
llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    temperature=0
)

# Connect to the embeddings model — turns text into a list of numbers for search
embeddings = AzureOpenAIEmbeddings(
    azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
)

# Connect to Azure AI Search — the index where your documents are stored
search_client = SearchClient(
    endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
    index_name="documents-index",
    credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_KEY"))
)

# Once the conversation exceeds this many messages, older ones get summarised
MAX_MESSAGES = 20


# =============================================================================
# STATE — the shared notepad all four nodes read from and write to
# The key difference from previous scripts: messages uses Annotated[List, operator.add]
# This means every time a node writes to messages, the new items are ADDED
# to the existing list rather than replacing it — this is how history builds up
# =============================================================================
class ConversationState(TypedDict):
    messages:       Annotated[List[dict], operator.add]  # every message in the conversation so far
    current_query:  str        # the question the user just asked
    resolved_query: str        # the same question rewritten to be self-contained
    context_docs:   List[dict] # documents retrieved from Azure AI Search for this turn
    response:       str        # the answer generated for this turn
    session_id:     str        # unique ID that ties all turns together in MemorySaver


# =============================================================================
# NODE 1 — Resolve Context: Fix follow-up questions before searching
# A follow-up like "When is it due?" cannot be searched directly.
# This node looks at the conversation history and rewrites it as a complete
# question: "When is the Contoso Consulting invoice INV-2024-0892 due?"
# First messages are passed through unchanged — no rewriting needed.
# =============================================================================
def resolve_context(state: ConversationState) -> dict:
    messages = state.get("messages", [])
    query = state["current_query"]

    # First turn of the conversation — nothing to resolve, pass straight through
    if len(messages) <= 1:
        print(f"[Context] First message — no resolution needed")
        return {"resolved_query": query}

    # Take the last 6 messages as context — enough history without overloading the prompt
    recent = messages[-6:] if len(messages) > 6 else messages

    # Format conversation history as readable text for the prompt
    history_text = "\n".join([
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
        for m in recent
    ])

    # Ask GPT-4o to make the question self-contained using what was discussed
    prompt = f"""Given this conversation history:
{history_text}

Rewrite the following question as a complete, self-contained question.
Replace pronouns and references with the actual entity names from the conversation.
Return only the rewritten question.

Question: {query}

Rewritten:"""

    result = llm.invoke(prompt)
    resolved = result.content.strip()
    print(f"[Context] Resolved: {resolved}")

    # Store the rewritten question — Node 2 will search with this instead
    return {"resolved_query": resolved}


# =============================================================================
# NODE 2 — Retrieve: Search Azure AI Search for relevant documents
# Uses the resolved (self-contained) query for best results.
# Runs hybrid search — keyword + vector similarity combined.
# Fetches 3 documents per turn (enough context, not too many tokens).
# =============================================================================
def retrieve_documents(state: ConversationState) -> dict:
    # Use the resolved query if available, fall back to the original question
    query = state.get("resolved_query") or state["current_query"]
    print(f"[Retrieve] Searching: {query}")

    # Convert the query into a vector for similarity search
    vector = embeddings.embed_query(query)
    vec_query = VectorizedQuery(vector=vector, k_nearest_neighbors=3, fields="content_vector")

    # Run hybrid search — finds both exact keyword matches and semantically similar documents
    results = search_client.search(
        search_text=query,
        vector_queries=[vec_query],
        top=3,
        select=["id", "content", "source_file", "document_type"]
    )

    # Convert results into a clean list of dictionaries
    docs = [{"content": r["content"], "source_file": r["source_file"],
             "document_type": r["document_type"]} for r in results]
    print(f"[Retrieve] Found {len(docs)} documents")
    return {"context_docs": docs}


# =============================================================================
# NODE 3 — Generate: Write a conversational, context-aware answer
# This node is different from the Day 2 RAG pipeline because it passes BOTH
# the retrieved documents AND the conversation history to GPT-4o.
# This lets the model say things like "As I mentioned earlier, the invoice..."
# making the conversation feel natural rather than starting fresh each time.
# =============================================================================
def generate_response(state: ConversationState) -> dict:
    docs = state.get("context_docs", [])
    print(f"[Generate] Generating response with {len(docs)} documents")

    messages = state.get("messages", [])

    # Include only the last 10 messages to avoid sending too many tokens to GPT-4o
    history_text = "\n".join([
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
        for m in messages[-10:]
    ])

    # Format the retrieved documents with source labels for citation
    if docs:
        context = "\n\n".join([f"[SOURCE {i+1}: {d['source_file']}]\n{d['content']}"
                                for i, d in enumerate(docs)])
    else:
        context = "No relevant documents found."

    # The prompt includes both conversation history and retrieved documents
    prompt = f"""You are a helpful document assistant for a financial services and
healthcare organisation. You have access to the conversation history and relevant
source documents.

Conversation history:
{history_text}

Relevant documents:
{context}

Answer the latest question. Use the documents as your primary source.
Cite sources with [SOURCE N]. Be conversational and aware of the history.

Latest question: {state["current_query"]}

Answer:"""

    result = llm.invoke(prompt)
    return {"response": result.content}


# =============================================================================
# NODE 4 — Update Memory: Save this turn to the conversation history
# Appends the current question and answer as two new messages.
# If the history is getting too long (over MAX_MESSAGES), it summarises the
# older messages into one summary message to keep the token count manageable.
# =============================================================================
def update_memory(state: ConversationState) -> dict:
    # Package the current question and answer as two new history entries
    new_messages = [
        {"role": "user",      "content": state["current_query"]},
        {"role": "assistant", "content": state["response"]}
    ]

    # Combine existing history with the two new messages
    all_messages = state.get("messages", []) + new_messages

    if len(all_messages) > MAX_MESSAGES:
        print(f"[Memory] Conversation too long ({len(all_messages)} messages) — summarising...")

        # Split into older messages to summarise and recent ones to keep in full
        older = all_messages[:-10]
        recent = all_messages[-10:]

        # Ask GPT-4o to summarise the older part of the conversation
        summary_prompt = f"""Summarise this conversation history in 3-4 sentences.
Focus on key facts discussed: document names, amounts, dates, decisions.

{chr(10).join([m["content"] for m in older])}

Summary:"""
        summary = llm.invoke(summary_prompt).content

        # Replace the older messages with one summary message plus the recent messages
        new_messages = [{"role": "system", "content": f"Earlier conversation summary: {summary}"}] + recent
        return {"messages": new_messages}

    # History is within limit — just append the two new messages
    return {"messages": new_messages}


# =============================================================================
# BUILD THE GRAPH — wire all four nodes in sequence
# =============================================================================
graph = StateGraph(ConversationState)

# Register all four nodes
graph.add_node("resolve_context",    resolve_context)
graph.add_node("retrieve_documents", retrieve_documents)
graph.add_node("generate_response",  generate_response)
graph.add_node("update_memory",      update_memory)

# Every conversation turn follows the same fixed path — no conditional routing needed
graph.set_entry_point("resolve_context")
graph.add_edge("resolve_context",    "retrieve_documents")
graph.add_edge("retrieve_documents", "generate_response")
graph.add_edge("generate_response",  "update_memory")
graph.add_edge("update_memory",      END)

# MemorySaver stores the full state after every turn
# The same session_id on the next call automatically restores the conversation
memory = MemorySaver()
assistant = graph.compile(checkpointer=memory)


# =============================================================================
# CHAT FUNCTION — the simple interface for sending one message at a time
# Each call with the same session_id continues the same conversation.
# Each call with a different session_id starts a fresh conversation.
# =============================================================================
def chat(question, session_id="session-001"):
    """Send one message and get a response. Maintains conversation history."""

    # The thread_id tells MemorySaver which conversation to load and save
    config = {"configurable": {"thread_id": session_id}}

    result = assistant.invoke({
        "messages":       [],   # MemorySaver restores the real history automatically
        "current_query":  question,
        "resolved_query": "",
        "context_docs":   [],
        "response":       "",
        "session_id":     session_id,
    }, config)

    return result["response"]


# =============================================================================
# RUN A DEMO CONVERSATION — five questions across two topics
# Notice how questions 2, 3, and 5 use "it" and "she" — pronouns that only
# make sense in the context of the earlier answers. The assistant resolves them.
# =============================================================================
if __name__ == "__main__":
    print("=== Conversational Document Assistant ===")
    print("Type your questions. The assistant remembers the conversation.")
    print("Type 'quit' to exit.\n")

    session = "demo-session-001"

    questions = [
        "What invoice do we have from Contoso Consulting?",
        "What is the total amount on it?",           # "it" refers to the Contoso invoice
        "When is it due?",                           # "it" still refers to the invoice
        "What medications is Sarah Johnson taking?", # new topic — switches to healthcare
        "Does she have any allergies?",              # "she" refers to Sarah Johnson
    ]

    for q in questions:
        print(f"\nYou: {q}")
        response = chat(q, session)
        print(f"Assistant: {response}")
