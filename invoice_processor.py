# =============================================================================
# PURPOSE
# This script extracts structured data from invoice documents using Azure AI
# Document Intelligence. It accepts invoices as URLs or local files, pulls out
# all key fields (vendor, total, dates, line items), flags low-confidence
# results for human review, and saves everything to invoice_results.json.
# =============================================================================


# --- Standard Python libraries ---
# os: reads environment variables (API key, endpoint) from the .env file
import os

# json: saves the extracted invoice data to a .json file at the end
import json

# datetime, timezone: stamps each extraction with the exact UTC time it ran
from datetime import datetime, timezone

# --- Third-party libraries (installed via pip) ---
# load_dotenv: reads your .env file and loads AZURE_* variables into memory
from dotenv import load_dotenv

# DocumentIntelligenceClient: the main Azure SDK class that talks to the service
from azure.ai.documentintelligence import DocumentIntelligenceClient

# AnalyzeDocumentRequest: wraps the document source (URL or bytes) into the
# format the Azure API expects
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest

# AzureKeyCredential: packages your API key into a secure credential object
from azure.core.credentials import AzureKeyCredential

# HttpResponseError: specific error raised when Azure returns a non-200 response
from azure.core.exceptions import HttpResponseError


# --- Load credentials from .env file into environment ---
# After this line, os.getenv() can read AZURE_* values from your .env file
load_dotenv()


# --- Create the Azure client ---
# This is the single connection object reused by every function in this script.
# endpoint: the URL of your Azure Document Intelligence resource
# credential: your API key wrapped in a secure credential object
client = DocumentIntelligenceClient(
    endpoint=os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT"),
    credential=AzureKeyCredential(os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY"))
)


# =============================================================================
# HELPER FUNCTION 1: get_field
# Safely reads one named field from the extracted document fields dictionary.
# Returns a dict with two keys: "value" (the text) and "confidence" (0.0-1.0).
# If the field is missing or errors, returns None for both — script never crashes.
# =============================================================================
def get_field(fields, name):
    try:
        # Field not present in this document — return empty result
        if name not in fields or fields[name] is None:
            return {"value": None, "confidence": None}

        f = fields[name]

        # Return the text content and round confidence to 4 decimal places
        return {
            "value": f.content,
            "confidence": round(f.confidence, 4) if f.confidence else None
        }
    except Exception:
        # Any unexpected error — return safely instead of crashing
        return {"value": None, "confidence": None}


# =============================================================================
# HELPER FUNCTION 2: extract_line_items
# Reads the "Items" array from invoice fields — this is the list of products
# or services billed on the invoice (e.g. "Consulting Hours x 10 @ £500").
# Returns a list of dicts, one per line item. Returns empty list if none found.
# Note: line item structure varies by SDK version — wrapped in try/except.
# =============================================================================
def extract_line_items(fields):
    try:
        # Get the Items field — this holds the array of line items
        items_field = fields.get("Items")

        # No Items field on this invoice — return empty list
        if items_field is None:
            return []

        # SDK stores array items in value_array — check it exists
        if not hasattr(items_field, "value_array"):
            return []

        raw = items_field.value_array

        # Items array is empty — nothing to extract
        if not raw:
            return []

        items = []
        for item in raw:
            # Each item stores its fields in value_object — skip if missing
            if not hasattr(item, "value_object") or not item.value_object:
                continue

            obj = item.value_object

            # Extract the four key fields from each line item
            items.append({
                "description": get_field(obj, "Description"),
                "quantity":    get_field(obj, "Quantity"),
                "unit_price":  get_field(obj, "UnitPrice"),
                "amount":      get_field(obj, "Amount"),
            })

        return items

    except Exception:
        # If line item parsing fails for any reason — return empty list safely
        return []


# =============================================================================
# CORE FUNCTION: analyze_invoice
# Sends a single invoice to Azure and returns all extracted fields as a list.
# source: either a URL string (is_url=True) or a local file path (is_url=False)
# Returns a list because one PDF can contain multiple invoices.
# =============================================================================
def analyze_invoice(source, is_url=True):
    try:
        if is_url:
            # Submit the document via its public URL — no download needed
            poller = client.begin_analyze_document(
                "prebuilt-invoice",                        # use the invoice model
                AnalyzeDocumentRequest(url_source=source)  # pass the URL
            )
        else:
            # Read the local file as raw bytes and submit to Azure
            with open(source, "rb") as f:
                data = f.read()
            poller = client.begin_analyze_document(
                "prebuilt-invoice",
                body=data,
                content_type="application/octet-stream"   # tells Azure it's a binary file
            )

        # Wait here until Azure finishes analysing — result holds all extracted data
        result = poller.result()

        invoices = []

        # Loop through each detected invoice in the document
        for doc in result.documents:
            fields = doc.fields

            invoices.append({
                # Timestamp of when this extraction ran (UTC)
                "extracted_at":     datetime.now(timezone.utc).isoformat(),

                # Overall model confidence for this document (0.0 to 1.0)
                "model_confidence": round(doc.confidence, 4),

                # All key invoice fields — each returns {value, confidence}
                "vendor_name":      get_field(fields, "VendorName"),
                "vendor_address":   get_field(fields, "VendorAddress"),
                "customer_name":    get_field(fields, "CustomerName"),
                "customer_address": get_field(fields, "CustomerAddress"),
                "invoice_id":       get_field(fields, "InvoiceId"),
                "invoice_date":     get_field(fields, "InvoiceDate"),
                "due_date":         get_field(fields, "DueDate"),
                "purchase_order":   get_field(fields, "PurchaseOrder"),
                "subtotal":         get_field(fields, "SubTotal"),
                "total_tax":        get_field(fields, "TotalTax"),
                "invoice_total":    get_field(fields, "InvoiceTotal"),
                "amount_due":       get_field(fields, "AmountDue"),

                # Array of line items (products/services billed)
                "line_items":       extract_line_items(fields),

                # Flag for human review — triggers if confidence drops below 85%
                "needs_review":     doc.confidence < 0.85,
            })

        return invoices

    except HttpResponseError as e:
        # Azure returned an error (wrong key, wrong endpoint, quota exceeded etc.)
        print(f"Azure API error {e.error.code}: {e.error.message}")
        raise

    except FileNotFoundError:
        # Local file path was wrong or file does not exist
        print(f"File not found: {source}")
        raise


# =============================================================================
# BATCH FUNCTION: process_batch
# Processes a list of invoice sources one by one.
# Collects successes in results[] and failures in errors[] separately.
# A single failure does not stop the rest of the batch from running.
# =============================================================================
def process_batch(sources):
    results, errors = [], []

    for idx, src in enumerate(sources):
        # Print progress so you know which document is being processed
        print(f"[{idx+1}/{len(sources)}] Processing: {src}")
        try:
            invoices = analyze_invoice(src)

            # Add all invoices from this document to the master results list
            results.extend(invoices)
            print(f"  OK - extracted {len(invoices)} invoice(s)")

        except Exception as e:
            # Log the failure and continue — do not stop the whole batch
            errors.append({"source": src, "error": str(e)})
            print(f"  FAILED: {e}")

    return results, errors


# =============================================================================
# ENTRY POINT
# This block runs only when you execute the script directly (python invoice_processor.py).
# It will not run if this file is imported as a module by another script (Day 4).
# =============================================================================
if __name__ == "__main__":

    # --- Define your invoice sources ---
    # Add more URLs or local file paths to this list to process multiple invoices
    sample_sources = [
        "https://raw.githubusercontent.com/Azure-Samples/"
        "cognitive-services-REST-api-samples/master/curl/"
        "form-recognizer/sample-invoice.pdf",
    ]

    print("=== Accounts Payable Invoice Processor ===")
    print(f"Documents to process: {len(sample_sources)}\n")

    # --- Run the batch ---
    results, errors = process_batch(sample_sources)

    # --- Print a human-readable summary to the terminal ---
    print("\n=== Extraction Summary ===")
    for i, inv in enumerate(results, 1):
        # Flag invoices that need a human to check them
        flag = " [NEEDS REVIEW]" if inv["needs_review"] else ""
        print(f"\nInvoice {i}{flag}")
        print(f"  Vendor    : {inv['vendor_name']['value']}")
        print(f"  Customer  : {inv['customer_name']['value']}")
        print(f"  Ref       : {inv['invoice_id']['value']}")
        print(f"  Total     : {inv['invoice_total']['value']}")
        print(f"  Line Items: {len(inv['line_items'])}")
        print(f"  Confidence: {inv['model_confidence']:.0%}")

    # --- Save full structured output to JSON ---
    # This file is used by indexer.py on Day 2 and by the agent tools on Day 4
    with open("invoice_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nFull results saved to invoice_results.json")

    # --- Report any failures ---
    if errors:
        print(f"\n{len(errors)} document(s) failed processing:")
        for err in errors:
            print(f"  {err['source']}: {err['error']}")


# =============================================================================
# SCRIPT SUMMARY
# =============================================================================
#
# WHAT THIS SCRIPT DOES
# Extracts structured data from invoice documents using the Azure AI Document
# Intelligence prebuilt-invoice model and saves the results to a JSON file.
#
# HOW IT WORKS — STEP BY STEP
# 1. load_dotenv()       reads your .env file to load the Azure key and endpoint
# 2. DocumentIntelligenceClient  creates one authenticated connection to Azure
# 3. analyze_invoice()   sends each invoice to Azure and waits for the result
# 4. get_field()         safely reads each named field (VendorName, Total etc.)
# 5. extract_line_items() pulls the array of products/services from the invoice
# 6. process_batch()     loops through all sources, handles failures gracefully
# 7. json.dump()         saves all extracted data to invoice_results.json
#
# KEY DESIGN DECISIONS
# - Every field read goes through get_field() so a missing field never crashes
# - needs_review flag is set automatically when confidence drops below 85%
# - Batch processing continues even if one invoice fails
# - if __name__ == "__main__" means this file can also be imported by Day 4
#   agent code without running the batch automatically
#
# OUTPUT FILES
# invoice_results.json — one JSON object per invoice with all extracted fields
#
# WHERE THIS IS USED NEXT
# Day 2: indexer.py reads invoice_results.json and loads it into Azure AI Search
# Day 4: the LangGraph document agent imports analyze_invoice() as a tool
# Day 5: the multi-agent capstone calls this module as part of the full pipeline
#
# COMMON ERRORS AND FIXES
# ValueError (missing endpoint/key)  → check your .env file has both values
# HttpResponseError 401              → your API key is wrong, re-copy from portal
# HttpResponseError 404              → your endpoint URL has a typo
# ModuleNotFoundError                → run: pip install azure-ai-documentintelligence
# =============================================================================