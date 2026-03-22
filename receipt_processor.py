# =============================================================================
# PURPOSE
# This script extracts structured data from receipt documents using Azure AI
# Document Intelligence. It accepts a receipt image or PDF via URL, pulls out
# key fields (merchant, total, tax, date), and saves the results to
# receipt_results.json for use in the Day 2 search index and Day 4 agent tools.
# =============================================================================


# --- Standard Python libraries ---
# os: reads environment variables (API key, endpoint) from the .env file
import os

# json: saves the extracted receipt data to a .json file at the end
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
# HELPER FUNCTION: safe_get
# Reads one named field from the extracted fields dictionary.
# Returns the text content if the field exists, or None if it is missing.
# Simpler than get_field() in invoice_processor because receipts do not need
# confidence scores per field — only the overall document confidence is used.
# =============================================================================
def safe_get(fields, name):
    # Look up the field by name — returns None if not present
    field = fields.get(name)

    # Field was not found on this receipt — return None safely
    if field is None:
        return None

    # Return just the text content of the field
    return field.content


# =============================================================================
# CORE FUNCTION: analyze_receipt
# Sends a receipt image or PDF to Azure and returns all extracted fields.
# url: publicly accessible URL pointing to the receipt file
# Returns a list because one file could theoretically contain multiple receipts.
# =============================================================================
def analyze_receipt(url):

    # Submit the receipt to Azure using the prebuilt-receipt model
    # prebuilt-receipt understands standard receipt layouts from any retailer
    poller = client.begin_analyze_document(
        "prebuilt-receipt",                       # use the receipt-specific model
        AnalyzeDocumentRequest(url_source=url)    # pass the receipt URL
    )

    # Wait here until Azure finishes analysing — result holds all extracted data
    result = poller.result()

    receipts = []

    # Loop through each detected receipt in the document
    for doc in result.documents:
        fields = doc.fields

        receipts.append({
            # Type of receipt: itemized, creditcard, gas, parking, hotel, generic
            "receipt_type":     safe_get(fields, "ReceiptType"),

            # Name of the business that issued the receipt
            "merchant_name":    safe_get(fields, "MerchantName"),

            # Physical address of the merchant
            "merchant_address": safe_get(fields, "MerchantAddress"),

            # Date the transaction took place
            "transaction_date": safe_get(fields, "TransactionDate"),

            # Time the transaction took place
            "transaction_time": safe_get(fields, "TransactionTime"),

            # Amount before tax
            "subtotal":         safe_get(fields, "Subtotal"),

            # Tax applied to the transaction
            "tax":              safe_get(fields, "Tax"),

            # Tip amount (common on restaurant and food receipts)
            "tip":              safe_get(fields, "Tip"),

            # Final total paid including tax and tip
            "total":            safe_get(fields, "Total"),

            # Overall model confidence for this receipt (0.0 to 1.0)
            "confidence":       round(doc.confidence, 4),
        })

    return receipts


# =============================================================================
# ENTRY POINT
# Runs only when you execute this script directly (python receipt_processor.py).
# Will not run if this file is imported as a module by another script (Day 4).
# =============================================================================
if __name__ == "__main__":

    # --- Define the receipt source ---
    # Replace this URL with any publicly accessible receipt image or PDF
    # For local files, modify analyze_receipt() to accept a file path instead
    sample_url = (
        "https://raw.githubusercontent.com/Azure-Samples/"
        "cognitive-services-REST-api-samples/master/curl/"
        "form-recognizer/contoso-receipt.png"
    )

    print("=== Receipt Processing Pipeline ===")

    # --- Send receipt to Azure and get extracted fields back ---
    receipts = analyze_receipt(sample_url)

    # --- Print a human-readable summary to the terminal ---
    for r in receipts:
        print(f"\nMerchant : {r['merchant_name']}")
        print(f"Date     : {r['transaction_date']}")
        print(f"Total    : {r['total']}")
        print(f"Tax      : {r['tax']}")
        print(f"Confidence: {r['confidence']:.0%}")

    # --- Save full structured output to JSON ---
    # This file is used by indexer.py on Day 2 and by the agent tools on Day 4
    with open("receipt_results.json", "w") as f:
        json.dump(receipts, f, indent=2)
    print("\nResults saved to receipt_results.json")


# =============================================================================
# SCRIPT SUMMARY
# =============================================================================
#
# WHAT THIS SCRIPT DOES
# Extracts structured data from receipt documents (images or PDFs) using the
# Azure AI Document Intelligence prebuilt-receipt model and saves results to
# receipt_results.json.
#
# HOW IT WORKS — STEP BY STEP
# 1. load_dotenv()       reads your .env file to load the Azure key and endpoint
# 2. DocumentIntelligenceClient  creates one authenticated connection to Azure
# 3. analyze_receipt()   sends the receipt URL to Azure and waits for the result
# 4. safe_get()          reads each named field safely — returns None if missing
# 5. json.dump()         saves all extracted data to receipt_results.json
#
# KEY DIFFERENCE FROM invoice_processor.py
# Receipts are simpler than invoices — no line items array, no batch processing,
# no needs_review flag. safe_get() returns just the text value without a
# confidence score per field because receipt fields are either present or not.
#
# SUPPORTED RECEIPT TYPES
# itemized, creditcard, gas, parking, hotel, generic
# The model auto-detects the type and returns it in the receipt_type field.
#
# OUTPUT FILES
# receipt_results.json — one JSON object per receipt with all extracted fields
#
# WHERE THIS IS USED NEXT
# Day 2: indexer.py reads receipt_results.json and loads it into Azure AI Search
# Day 4: the LangGraph document agent imports analyze_receipt() as a tool
# Day 5: the multi-agent capstone calls this module as part of the full pipeline
#
# COMMON ERRORS AND FIXES
# ValueError (missing endpoint/key)  → check your .env file has both values
# HttpResponseError 401              → your API key is wrong, re-copy from portal
# HttpResponseError 404              → your endpoint URL has a typo
# No output / empty receipts list    → the URL may be unreachable, check internet
# ModuleNotFoundError                → run: pip install azure-ai-documentintelligence
# =============================================================================
