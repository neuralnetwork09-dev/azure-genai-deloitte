# =============================================================================
# PURPOSE
# This script builds the first fully autonomous AI agent of the program.
# You give it a request in plain English. It reads the request, decides which
# tool to use, runs the tool, reads the result, and gives you a final answer.
# You write zero routing logic — GPT-4o makes every decision on its own.
# This becomes the document extraction specialist agent in the Day 4 system.
# =============================================================================

# os: reads your API keys and endpoints from the .env file
import os

# json: not used directly here but available for any JSON handling needed
import json

# TypedDict: defines the shape of the State
# List: declares a field that holds a list
# Annotated: adds special behaviour — used here to make messages accumulate
from typing import TypedDict, List, Annotated

# operator.add: tells LangGraph to APPEND new messages rather than replace them
import operator

# load_dotenv: reads the .env file so credentials are available via os.getenv()
from dotenv import load_dotenv

# StateGraph: builds the workflow graph | END: marks where the workflow finishes
from langgraph.graph import StateGraph, END

# ToolNode: a pre-built LangGraph node that automatically runs whichever tool
# GPT-4o selected — you do not have to write the tool execution logic yourself
from langgraph.prebuilt import ToolNode

# AzureChatOpenAI: connects to GPT-4o for reasoning and generating answers
# AzureOpenAIEmbeddings: converts text into vectors for similarity search
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

# @tool decorator: wraps a Python function so GPT-4o can call it as a tool
# The docstring of each decorated function becomes GPT-4o's instruction manual
from langchain_core.tools import tool

# SearchClient: used to query the Azure AI Search index
from azure.search.documents import SearchClient

# VectorizedQuery: wraps a vector so Azure AI Search can find similar documents
from azure.search.documents.models import VectorizedQuery

# AzureKeyCredential: securely passes your API key to Azure services
from azure.core.credentials import AzureKeyCredential

# Import the Day 1 processor functions — the agent will call these as tools
# These are the same files you built in Lab 1.2 — they are being reused here
from invoice_processor import analyze_invoice
from layout_processor  import analyze_layout

# Load all credentials from the .env file into memory
load_dotenv()

# Connect to GPT-4o — this is the brain that reads requests and picks tools
# temperature=0 ensures the agent makes consistent, predictable decisions
llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    temperature=0
)

# Connect to the embeddings model — converts search queries into vectors
embeddings = AzureOpenAIEmbeddings(
    azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
)

# Connect to Azure AI Search — the index where your Day 2 documents are stored
search_client = SearchClient(
    endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
    index_name="documents-index",
    credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_KEY"))
)


# =============================================================================
# TOOL 1 — Extract Invoice Data
# Wraps the analyze_invoice() function from Day 1 as an agent tool.
# The @tool decorator and the docstring are what GPT-4o reads to decide
# whether to call this tool. The clearer the docstring, the smarter the agent.
# =============================================================================
@tool
def extract_invoice_data(document_url: str) -> str:
    """
    Extract structured data from an invoice document.
    Use this tool when the user wants to process, analyse, or extract fields from an invoice.
    Returns vendor name, invoice number, dates, totals, and line items.
    Input: document_url — a publicly accessible URL pointing to the invoice PDF or image.
    """
    print(f"[Tool: extract_invoice_data] Processing: {document_url}")
    try:
        # Call the Day 1 analyze_invoice function — same code, now called by an agent
        results = analyze_invoice(document_url, is_url=True)

        if not results:
            return "No invoice data could be extracted from this document."

        # Take the first invoice result and format it as readable text for the agent
        inv = results[0]
        summary = (
            f"Invoice extracted successfully.\n"
            f"Vendor: {inv['vendor_name']['value']}\n"
            f"Invoice ID: {inv['invoice_id']['value']}\n"
            f"Date: {inv['invoice_date']['value']}\n"
            f"Due Date: {inv['due_date']['value']}\n"
            f"Total: {inv['invoice_total']['value']}\n"
            f"Confidence: {inv['model_confidence']:.0%}\n"
            f"Needs Review: {inv['needs_review']}"
        )
        return summary

    except Exception as e:
        # Return the error as a string — the agent reads this and handles it gracefully
        return f"Error extracting invoice: {str(e)}"


# =============================================================================
# TOOL 2 — Analyse Document Layout
# Wraps the analyze_layout() function from Day 1 as an agent tool.
# Used for any document that is NOT an invoice — contracts, reports, forms etc.
# The docstring tells GPT-4o exactly when to prefer this tool over Tool 1.
# =============================================================================
@tool
def analyse_document_layout(document_url: str) -> str:
    """
    Analyse the layout and structure of any document.
    Use this tool when the user wants to understand the structure of a document,
    extract tables, count pages, or process a document that is not an invoice or receipt.
    Works with contracts, reports, forms, clinical notes, and any other document type.
    Input: document_url — a publicly accessible URL pointing to the document.
    """
    print(f"[Tool: analyse_document_layout] Processing: {document_url}")
    try:
        # Call the Day 1 analyze_layout function
        layout = analyze_layout(document_url)

        # Build a readable summary of the layout for the agent to work with
        summary = (
            f"Layout analysis complete.\n"
            f"Pages: {layout['page_count']}\n"
            f"Tables detected: {layout['table_count']}\n"
            f"Total words: {sum(p['word_count'] for p in layout['pages'])}\n"
        )

        # Add table details if any were found
        if layout["tables"]:
            summary += f"\nTable summary:\n"
            for t in layout["tables"][:3]:  # show first 3 tables only
                summary += f"  Table {t['table_index']}: {t['row_count']} rows x {t['column_count']} columns\n"

        # Add a preview of the extracted text
        summary += f"\nFull text preview:\n{layout['full_text'][:500]}..."
        return summary

    except Exception as e:
        return f"Error analysing layout: {str(e)}"


# =============================================================================
# TOOL 3 — Search Knowledge Base
# Searches the Azure AI Search index using the query as both keyword and vector.
# Used when the user asks about documents that have already been indexed —
# i.e. the documents loaded by indexer.py in Day 2.
# =============================================================================
@tool
def search_knowledge_base(query: str) -> str:
    """
    Search the enterprise knowledge base for information about documents.
    Use this tool when the user asks a question about previously processed documents,
    wants to find specific information across multiple documents, or asks about
    invoices, receipts, or clinical documents that have already been indexed.
    Input: query — the search question in natural language.
    """
    print(f"[Tool: search_knowledge_base] Searching: {query}")
    try:
        # Convert the query into a vector for similarity search
        vector = embeddings.embed_query(query)
        vec_query = VectorizedQuery(vector=vector, k_nearest_neighbors=3, fields="content_vector")

        # Run hybrid search — keyword + vector combined
        results = search_client.search(
            search_text=query,
            vector_queries=[vec_query],
            top=3,
            select=["content", "source_file", "document_type"]
        )

        docs = list(results)

        if not docs:
            return "No relevant documents found in the knowledge base."

        # Format results as readable text — the agent uses this to write its answer
        output = f"Found {len(docs)} relevant documents:\n\n"
        for i, doc in enumerate(docs, 1):
            output += f"[{i}] {doc['source_file']} ({doc['document_type']})\n"
            output += f"    {doc['content'][:300]}\n\n"
        return output

    except Exception as e:
        return f"Error searching knowledge base: {str(e)}"


# =============================================================================
# AGENT STATE — the shared notepad for this agent
# Only one field: messages — the full list of everything that has happened
# including the user request, tool calls, tool results, and final answers.
# operator.add means every new message is appended, not replacing the list.
# =============================================================================
class AgentState(TypedDict):
    messages: Annotated[List, operator.add]  # accumulates all messages and tool results


# =============================================================================
# BIND TOOLS TO THE MODEL
# This is the step that gives GPT-4o its powers.
# bind_tools() sends the three tool descriptions to GPT-4o alongside every request.
# GPT-4o reads those descriptions and decides which tool fits the user's request.
# =============================================================================
tools = [extract_invoice_data, analyse_document_layout, search_knowledge_base]
llm_with_tools = llm.bind_tools(tools)


# =============================================================================
# AGENT NODE — the reasoning step
# This is where GPT-4o thinks. It reads all the messages so far and either:
# Option A: decides to call a tool — returns a tool_call message
# Option B: has enough information to answer — returns a final text message
# The routing function below checks which option was chosen.
# =============================================================================
def agent_node(state: AgentState) -> dict:
    print("[Agent] Reasoning...")

    # Pass the full message history to GPT-4o with the tool descriptions attached
    response = llm_with_tools.invoke(state["messages"])

    # Append GPT-4o's response to the messages list — operator.add handles this
    return {"messages": [response]}


# =============================================================================
# ROUTING FUNCTION — should we run a tool or stop?
# Reads the last message in state to check what GPT-4o decided.
# If GPT-4o wants to call a tool, return "tools" — LangGraph routes to ToolNode.
# If GPT-4o wrote a final answer, return END — the workflow finishes.
# =============================================================================
def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]

    # tool_calls is set by GPT-4o when it wants to run a tool
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        print(f"[Route] Calling tool: {last_message.tool_calls[0]['name']}")
        return "tools"   # go to ToolNode — it will run the selected tool

    # No tool call — GPT-4o has written a final answer
    print("[Route] Agent has final answer — ending workflow")
    return END


# =============================================================================
# BUILD THE AGENT GRAPH
# Only two nodes: agent (reasoning) and tools (execution).
# The loop between them is what makes this an agent rather than a pipeline:
# agent reasons -> tool runs -> agent reads result -> reasons again -> answer
# =============================================================================
graph = StateGraph(AgentState)

# The agent node is where all the thinking happens
graph.add_node("agent", agent_node)

# ToolNode automatically runs whichever tool GPT-4o selected — no code needed
graph.add_node("tools", ToolNode(tools))

# Every request starts with the agent reasoning about what to do
graph.set_entry_point("agent")

# After the agent reasons, check: tool call or final answer?
graph.add_conditional_edges("agent", should_continue)

# After a tool runs, send the result back to the agent to reason again
graph.add_edge("tools", "agent")

# Lock the graph into a runnable application
document_agent = graph.compile()


# =============================================================================
# ASK AGENT — the simple interface for sending one request at a time
# Wraps the question in a HumanMessage and runs the full agent graph.
# The last message in the result is always the final answer.
# =============================================================================
def ask_agent(question):
    """Send a question to the agent and get a response."""
    from langchain_core.messages import HumanMessage

    result = document_agent.invoke({
        "messages": [HumanMessage(content=question)]
    })

    # The final answer is always the last message in the list
    return result["messages"][-1].content


# =============================================================================
# RUN THREE TEST REQUESTS — each should trigger a different tool
# Watch the [Route] and [Tool] lines in the terminal to see the agent deciding
# =============================================================================
if __name__ == "__main__":
    print("=== Autonomous Document Processing Agent ===")
    print("The agent will decide which tool to use for each request.\n")

    test_requests = [
        # Contains "invoice" and a URL — should trigger extract_invoice_data
        "Extract the invoice fields from this document: https://raw.githubusercontent.com/Azure-Samples/cognitive-services-REST-api-samples/master/curl/form-recognizer/sample-invoice.pdf",

        # Asks about indexed documents — should trigger search_knowledge_base
        "What do we know about Contoso Consulting from our indexed documents?",

        # Asks about structure and layout — should trigger analyse_document_layout
        "Analyse the structure and layout of this document: https://raw.githubusercontent.com/Azure-Samples/cognitive-services-REST-api-samples/master/curl/form-recognizer/sample-invoice.pdf",
    ]

    for request in test_requests:
        print(f"\nRequest: {request[:80]}...")
        response = ask_agent(request)
        print(f"Agent: {response}")
        print("-" * 60)