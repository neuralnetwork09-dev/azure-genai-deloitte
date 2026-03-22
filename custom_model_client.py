
# =============================================================================
# PURPOSE
# This script tests your trained custom model against real documents. It sends
# a document to your custom model, displays every extracted field with its
# confidence score, and can evaluate model accuracy across a test set of
# multiple documents. Run this after training completes in Studio to verify
# your model is working correctly before using it in the Day 4 agent.
# =============================================================================


# --- Standard Python libraries ---
# os: reads environment variables (API key, endpoint) from the .env file
import os

# json: saves the extracted fields to a .json file at the end
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
# Single connection object reused by every function in this script
client = DocumentIntelligenceClient(
    endpoint=os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT"),
    credential=AzureKeyCredential(os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY"))
)


# --- Your custom model ID ---
# This must exactly match the Model ID you gave when training in Studio
# Run list_models.py if you are unsure what your model ID is
CUSTOM_MODEL_ID = "purchase-order-model-v2"


# =============================================================================
# CORE FUNCTION: analyze_with_custom_model
# Sends a document to your custom trained model and returns all extracted
# fields with their values and confidence scores.
# url: publicly accessible URL pointing to the document
# model_id: the ID of your custom model (defaults to CUSTOM_MODEL_ID above)
# Returns a list of dicts — one per document detected in the file.
#
# KEY DIFFERENCE FROM pre-built processors:
# Pre-built models return fixed known fields (VendorName, Total etc.).
# Your custom model returns the exact field names YOU defined when labeling
# in Document Intelligence Studio (e.g. purchase_order_number, vendor_name).
# =============================================================================
def analyze_with_custom_model(url, model_id=CUSTOM_MODEL_ID):

    # Submit the document to Azure using YOUR custom trained model
    poller = client.begin_analyze_document(
        model_id,                                 # your custom model ID
        AnalyzeDocumentRequest(url_source=url)    # document URL to analyse
    )

    # Wait here until Azure finishes — result holds all extracted field data
    result = poller.result()

    all_fields = []

    # Loop through each detected document in the file
    for doc in result.documents:

        # Build the top-level result for this document
        doc_fields = {
            # The model ID that matched this document (useful for composed models)
            "doc_type":   doc.doc_type,

            # Overall confidence the model has in this extraction (0.0 to 1.0)
            "confidence": round(doc.confidence, 4),

            # Individual field results — populated in the loop below
            "fields":     {}
        }

        # --- Extract every field your custom model was trained to find ---
        for field_name, field_value in doc.fields.items():

            if field_value is None:
                # Field was not found on this document
                doc_fields["fields"][field_name] = {"value": None, "confidence": None}
            else:
                # Field was found — store text content and confidence score
                doc_fields["fields"][field_name] = {
                    "value":      field_value.content,
                    "confidence": round(field_value.confidence, 4) if field_value.confidence else None
                }

        all_fields.append(doc_fields)

    return all_fields


# =============================================================================
# EVALUATION FUNCTION: evaluate_model_on_test_set
# Tests your custom model against a list of document URLs and reports the
# average confidence score per field across all test documents.
# Use this to identify which fields need more training data.
# Fields scoring below 80% are flagged as REVIEW NEEDED.
# =============================================================================
def evaluate_model_on_test_set(test_urls, model_id=CUSTOM_MODEL_ID):

    # Dictionary to collect all confidence scores per field across all documents
    field_confidences = {}

    # Counter for how many documents were successfully tested
    doc_count = 0

    for url in test_urls:
        try:
            results = analyze_with_custom_model(url, model_id)

            for doc in results:
                doc_count += 1

                # Collect confidence scores for each field in this document
                for fname, fdata in doc["fields"].items():
                    if fdata["confidence"] is not None:

                        # Create a list for this field if first time seeing it
                        if fname not in field_confidences:
                            field_confidences[fname] = []

                        # Add this document's confidence score to the list
                        field_confidences[fname].append(fdata["confidence"])

        except Exception as e:
            # Log the failure and continue testing remaining URLs
            print(f"  Error on {url}: {e}")

    # --- Print evaluation report ---
    print(f"\nEvaluation complete. Documents tested: {doc_count}")
    print("\nAverage confidence per field:")

    # Sort fields alphabetically for easy reading
    for fname, scores in sorted(field_confidences.items()):

        # Calculate average confidence across all test documents for this field
        avg = sum(scores) / len(scores)

        # Flag fields below 80% — these need more labeled training samples
        status = "OK" if avg >= 0.80 else "REVIEW NEEDED"

        # Print field name, average confidence, and status
        print(f"  {fname:<30} {avg:.0%}  [{status}]")


# =============================================================================
# ENTRY POINT
# Runs only when you execute this script directly (python custom_model_client.py)
# Will not run if this file is imported as a module by another script (Day 4).
# =============================================================================
if __name__ == "__main__":

    # --- Define the test document ---
    # This is one of the five training forms used in Lab 1.3
    # Replace with any document your custom model was trained to handle
    test_url = (
        "https://raw.githubusercontent.com/Azure/azure-sdk-for-python/main/sdk/"
        "formrecognizer/azure-ai-formrecognizer/samples/sample_forms/training/Form_1.jpg"
    )

    print(f"Testing custom model: {CUSTOM_MODEL_ID}")

    # --- Send document to Azure and get extracted fields back ---
    results = analyze_with_custom_model(test_url)

    # --- Print a detailed field-by-field report to the terminal ---
    for doc in results:
        print(f"\nDocument type : {doc['doc_type']}")
        print(f"Confidence    : {doc['confidence']:.0%}")
        print("Fields:")

        for fname, fdata in doc["fields"].items():
            # Format confidence as percentage or N/A if not available
            conf_str = f"{fdata['confidence']:.0%}" if fdata["confidence"] else "N/A"

            # Print field name (left-aligned 30 chars), value (40 chars), confidence
            print(f"  {fname:<30} {str(fdata['value']):<40} conf: {conf_str}")

    # --- Save full structured output to JSON ---
    # This file is used by the Day 4 agent to verify extraction quality
    with open("custom_model_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to custom_model_results.json")


# =============================================================================
# SCRIPT SUMMARY
# =============================================================================
#
# WHAT THIS SCRIPT DOES
# Tests your custom trained Azure AI Document Intelligence model against real
# documents. Extracts all labeled fields with confidence scores, prints a
# field-by-field report, optionally evaluates accuracy across a test set, and
# saves results to custom_model_results.json.
#
# HOW IT WORKS — STEP BY STEP
# 1. load_dotenv()                  reads your .env file for key and endpoint
# 2. CUSTOM_MODEL_ID                set this to your exact model ID from Studio
# 3. analyze_with_custom_model()    sends document to YOUR model, not a pre-built
# 4. doc.fields.items()             loops through every field YOUR model extracts
# 5. evaluate_model_on_test_set()   optional — tests multiple docs, reports accuracy
# 6. json.dump()                    saves all results to custom_model_results.json
#
# TWO FUNCTIONS EXPLAINED
# analyze_with_custom_model()   test ONE document — use this daily during labs
# evaluate_model_on_test_set()  test MULTIPLE documents — use this to check if
#                               your model needs more training data
#
# UNDERSTANDING THE OUTPUT
# doc_type      the model ID that handled this document
# confidence    overall model confidence — below 85% triggers needs_review
# fields{}      every field you labeled in Studio with value and confidence
# [OK]          field confidence above 80% — acceptable for production
# [REVIEW NEEDED] field confidence below 80% — add more labeled samples
#
# HOW TO IMPROVE A LOW-CONFIDENCE FIELD
# 1. Open Document Intelligence Studio
# 2. Add 10-20 more labeled samples for that specific field
# 3. Retrain the model
# 4. Run this script again and compare scores
#
# OUTPUT FILES
# custom_model_results.json — all extracted fields with values and confidence
#
# WHERE THIS IS USED NEXT
# Day 4: the LangGraph document agent imports analyze_with_custom_model() as
#        a tool — the agent calls it automatically when it receives a
#        purchase order document that does not match any pre-built model
# Day 5: the multi-agent capstone uses this as the custom extraction tool
#        in the document routing pipeline
#
# COMMON ERRORS AND FIXES
# ValueError (missing endpoint/key)    → check your .env file has both values
# HttpResponseError 401                → your API key is wrong, re-copy from portal
# HttpResponseError 404 on model       → CUSTOM_MODEL_ID does not match Studio name
# ModelNotReady error                  → training still running, wait and retry
# Empty fields dict                    → document type does not match your model
# ModuleNotFoundError                  → run: pip install azure-ai-documentintelligence
# =============================================================================
