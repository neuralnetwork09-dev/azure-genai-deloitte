# =============================================================================
# PURPOSE
# This script analyses the full structure of a document using Azure AI Document
# Intelligence. Unlike invoice_processor.py which extracts named fields, this
# script maps everything — pages, dimensions, word counts, and complete table
# structures with row and column data. It is the general-purpose extraction tool
# used when a document type has no matching pre-built model.
# =============================================================================


# --- Standard Python libraries ---
# os: reads environment variables (API key, endpoint) from the .env file
import os

# json: saves the full layout structure to a .json file at the end
import json

# --- Third-party libraries (installed via pip) ---
# load_dotenv: reads your .env file and loads AZURE_* variables into memory
from dotenv import load_dotenv

# DocumentIntelligenceClient: the main Azure SDK class that talks to the service
from azure.ai.documentintelligence import DocumentIntelligenceClient

# AnalyzeDocumentRequest: wraps the document URL into the format Azure expects
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest

# AzureKeyCredential: packages your API key into a secure credential object
from azure.core.credentials import AzureKeyCredential


# --- Load credentials from .env file into environment ---
# After this line, os.getenv() can read AZURE_* values from your .env file
load_dotenv()


# --- Create the Azure client ---
# Single connection object reused by every function in this script.
# endpoint: the URL of your Azure Document Intelligence resource
# credential: your API key wrapped in a secure credential object
client = DocumentIntelligenceClient(
    endpoint=os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT"),
    credential=AzureKeyCredential(os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY"))
)


# =============================================================================
# CORE FUNCTION: analyze_layout
# Sends a document to Azure and returns its complete structural layout.
# url: publicly accessible URL pointing to the document (PDF or image)
# Returns a single dict containing page info, table data, and full text.
#
# KEY DIFFERENCE FROM invoice/receipt processors:
# prebuilt-layout does NOT extract named fields like VendorName or Total.
# Instead it maps the spatial structure — where are the tables, how many rows,
# what are the headings, what is the reading order. Think of it as an X-ray
# of the document structure rather than a field extractor.
# =============================================================================
def analyze_layout(url):

    # Submit the document to Azure using the prebuilt-layout model
    # prebuilt-layout works on any document type — no specific training needed
    poller = client.begin_analyze_document(
        "prebuilt-layout",                        # use the layout analysis model
        AnalyzeDocumentRequest(url_source=url)    # pass the document URL
    )

    # Wait here until Azure finishes analysing — result holds full layout data
    result = poller.result()

    # Build the output dictionary — this is what gets saved to JSON at the end
    output = {
        # Total number of pages in the document
        "page_count":  len(result.pages),

        # Total number of tables detected (0 if none found)
        "table_count": len(result.tables) if result.tables else 0,

        # Per-page details populated in the loop below
        "pages":       [],

        # Table structures populated in the loop below
        "tables":      [],

        # Complete extracted text of the entire document in reading order
        # This is the raw text used as input for the Day 2 RAG pipeline
        "full_text":   result.content
    }

    # --- Extract per-page information ---
    for page in result.pages:
        output["pages"].append({
            # Page number (starts at 1)
            "page_number": page.page_number,

            # Physical width of the page
            "width":       page.width,

            # Physical height of the page
            "height":      page.height,

            # Unit of measurement: "inch" or "pixel"
            "unit":        page.unit,

            # Number of individual words detected on this page
            "word_count":  len(page.words) if page.words else 0,

            # Number of text lines detected on this page
            "line_count":  len(page.lines) if page.lines else 0,
        })

    # --- Extract table structures (only if tables were detected) ---
    if result.tables:
        for t_idx, table in enumerate(result.tables):

            # Start with the table-level summary
            table_data = {
                # Zero-based index of this table in the document
                "table_index":  t_idx,

                # Total rows in this table
                "row_count":    table.row_count,

                # Total columns in this table
                "column_count": table.column_count,

                # Individual cell data populated in the loop below
                "cells":        []
            }

            # --- Extract every cell in this table ---
            for cell in table.cells:
                table_data["cells"].append({
                    # Row position of this cell (0-based)
                    "row_index":    cell.row_index,

                    # Column position of this cell (0-based)
                    "column_index": cell.column_index,

                    # How many rows this cell spans (merged cells > 1)
                    "row_span":     cell.row_span,

                    # How many columns this cell spans (merged cells > 1)
                    "column_span":  cell.column_span,

                    # The actual text content inside this cell
                    "content":      cell.content,

                    # Cell type: "content", "columnHeader", or "rowHeader"
                    "kind":         cell.kind
                })

            output["tables"].append(table_data)

    return output


# =============================================================================
# DISPLAY FUNCTION: print_table
# Renders an extracted table as a readable grid in the terminal.
# Builds a 2D list from the flat cells array, then prints row by row.
# Each cell is truncated to 25 characters and padded for alignment.
# =============================================================================
def print_table(table_data):

    # Print the table header showing its dimensions
    print(f"Table {table_data['table_index']}: {table_data['row_count']} rows x {table_data['column_count']} columns")

    # Build an empty 2D grid (rows x columns) filled with empty strings
    grid = [["" for _ in range(table_data["column_count"])]
            for _ in range(table_data["row_count"])]

    # Place each cell's content into the correct position in the grid
    for cell in table_data["cells"]:
        grid[cell["row_index"]][cell["column_index"]] = cell["content"]

    # Print each row — truncate long text at 25 chars, pad short text with spaces
    for row in grid:
        print("  | " + " | ".join(str(c)[:25].ljust(25) for c in row))


# =============================================================================
# ENTRY POINT
# Runs only when you execute this script directly (python layout_processor.py).
# Will not run if this file is imported as a module by another script (Day 4).
# =============================================================================
if __name__ == "__main__":

    # --- Define the document source ---
    # Replace this URL with any publicly accessible document URL
    # Works with: invoices, contracts, reports, forms, clinical documents
    sample_url = (
        "https://raw.githubusercontent.com/Azure-Samples/"
        "cognitive-services-REST-api-samples/master/curl/"
        "form-recognizer/sample-invoice.pdf"
    )

    print("=== Layout Analysis Pipeline ===")

    # --- Send document to Azure and get full layout back ---
    layout = analyze_layout(sample_url)

    # --- Print page-level summary ---
    print(f"Pages    : {layout['page_count']}")
    print(f"Tables   : {layout['table_count']}")
    for p in layout["pages"]:
        print(f"  Page {p['page_number']}: {p['word_count']} words, {p['line_count']} lines")

    # --- Print each detected table as a formatted grid ---
    if layout["tables"]:
        print("\nExtracted Tables:")
        for t in layout["tables"]:
            print_table(t)

    # --- Save full structured output to JSON ---
    # This file is used by indexer.py on Day 2 to chunk and index the full_text
    # The table structures are used by the agent on Day 4 for structured queries
    with open("layout_results.json", "w") as f:
        json.dump(layout, f, indent=2)
    print("\nResults saved to layout_results.json")


# =============================================================================
# SCRIPT SUMMARY
# =============================================================================
#
# WHAT THIS SCRIPT DOES
# Analyses the complete structural layout of any document using the Azure AI
# Document Intelligence prebuilt-layout model. Extracts page dimensions, word
# counts, full text in reading order, and complete table structures including
# every cell's row, column, span, and content. Saves everything to
# layout_results.json.
#
# HOW IT WORKS — STEP BY STEP
# 1. load_dotenv()       reads your .env file to load the Azure key and endpoint
# 2. DocumentIntelligenceClient  creates one authenticated connection to Azure
# 3. analyze_layout()    sends the document URL to Azure and waits for result
# 4. Page loop           extracts dimensions and word/line counts per page
# 5. Table loop          extracts every table with full row/column/cell data
# 6. print_table()       renders each table as a readable grid in the terminal
# 7. json.dump()         saves the complete layout structure to layout_results.json
#
# KEY DIFFERENCE FROM OTHER PROCESSORS
# prebuilt-layout does NOT extract named fields (no VendorName, no Total).
# It maps the document structure — pages, tables, reading order, spatial layout.
# Use this when: no pre-built model matches your document type, or when you need
# table data from a financial report, contract, or regulatory filing.
#
# WHAT IS IN layout_results.json
# page_count    total pages in the document
# table_count   total tables detected
# pages[]       per-page: number, width, height, unit, word_count, line_count
# tables[]      per-table: row_count, column_count, cells[]
# cells[]       per-cell: row_index, column_index, spans, content, kind
# full_text     complete document text in reading order (used by RAG pipeline)
#
# OUTPUT FILES
# layout_results.json — complete structural layout of the document
#
# WHERE THIS IS USED NEXT
# Day 2: indexer.py reads full_text from layout_results.json, chunks it, and
#        loads it into Azure AI Search as the document knowledge base
# Day 4: the LangGraph document agent imports analyze_layout() as a tool for
#        documents that do not match any pre-built or custom model
# Day 5: the multi-agent capstone uses this as the fallback extraction tool
#
# COMMON ERRORS AND FIXES
# ValueError (missing endpoint/key)  → check your .env file has both values
# HttpResponseError 401              → your API key is wrong, re-copy from portal
# HttpResponseError 404              → your endpoint URL has a typo
# Empty tables list                  → the document has no tables, this is normal
# ModuleNotFoundError                → run: pip install azure-ai-documentintelligence
# =============================================================================
