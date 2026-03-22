
# =============================================================================
# PURPOSE
# This utility script lists every model available in your Azure AI Document
# Intelligence resource — both Microsoft pre-built models and any custom models
# you have trained. Run this whenever you need to confirm a model ID before
# using it in code, or to verify your custom model training has completed.
# =============================================================================


# --- Standard Python libraries ---
# os: reads environment variables (API key, endpoint) from the .env file
import os

# --- Third-party libraries (installed via pip) ---
# load_dotenv: reads your .env file and loads AZURE_* variables into memory
from dotenv import load_dotenv

# DocumentIntelligenceAdministrationClient: the admin-side SDK client
# Different from DocumentIntelligenceClient used in other scripts —
# this one manages models (list, create, delete) rather than analysing documents
from azure.ai.documentintelligence import DocumentIntelligenceAdministrationClient

# AzureKeyCredential: packages your API key into a secure credential object
from azure.core.credentials import AzureKeyCredential


# --- Load credentials from .env file into environment ---
# After this line, os.getenv() can read AZURE_* values from your .env file
load_dotenv()


# --- Create the Azure administration client ---
# Note: this is the ADMIN client, not the regular DocumentIntelligenceClient.
# Use this only for model management tasks like listing, composing, or deleting.
# Use DocumentIntelligenceClient (in other scripts) for actual document analysis.
admin_client = DocumentIntelligenceAdministrationClient(
    endpoint=os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT"),
    credential=AzureKeyCredential(os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY"))
)


# --- Print all models in this resource ---
print("Models in your Document Intelligence resource:")
print("-" * 70)

# list_models() returns every model: all Microsoft pre-built models AND
# any custom models you have trained in this resource
for model in admin_client.list_models():

    # Classify the model as prebuilt or custom based on its ID prefix
    # All Microsoft models start with "prebuilt-" (e.g. prebuilt-invoice)
    # Your trained models use the name you gave them (e.g. purchase-order-model-v1)
    model_type = "prebuilt" if model.model_id.startswith("prebuilt") else "custom"

    # Print the model type tag and model ID on one line
    print(f"  [{model_type}] {model.model_id}")

    # Print the description on the next line if one exists
    # Pre-built models always have descriptions — custom models show yours
    if hasattr(model, "description") and model.description:
        print(f"             {model.description}")


# =============================================================================
# SCRIPT SUMMARY
# =============================================================================
#
# WHAT THIS SCRIPT DOES
# Connects to your Azure AI Document Intelligence resource using the admin
# client and prints a complete list of all available models — both Microsoft
# pre-built models and any custom models you have trained. Each model is
# labelled [prebuilt] or [custom] for easy identification.
#
# HOW IT WORKS — STEP BY STEP
# 1. load_dotenv()                  reads your .env file for key and endpoint
# 2. DocumentIntelligenceAdministrationClient  connects to the admin API
# 3. admin_client.list_models()     fetches every model in your resource
# 4. model_id.startswith("prebuilt")  classifies each model as prebuilt/custom
# 5. Prints model ID and description for every model found
#
# TWO CLIENTS EXPLAINED
# DocumentIntelligenceClient        — used to ANALYSE documents (all other scripts)
# DocumentIntelligenceAdministrationClient — used to MANAGE models (this script only)
# Both use the same endpoint and key from your .env file.
#
# WHEN TO RUN THIS SCRIPT
# After training a custom model  → confirm it appears as [custom] in the list
# Before writing agent code      → copy the exact model ID to use in your script
# After Day 1 Lab 1.3            → verify purchase-order-model-v1 shows as succeeded
# Any time you forget a model ID → run this instead of opening the Azure Portal
#
# WHAT TO LOOK FOR IN THE OUTPUT
# [prebuilt] prebuilt-invoice      → standard invoice extraction — used in Lab 1.2
# [prebuilt] prebuilt-receipt      → standard receipt extraction — used in Lab 1.2
# [prebuilt] prebuilt-layout       → layout analysis — used in Lab 1.2
# [prebuilt] prebuilt-read         → text only — used in test_connection.py
# [custom]   purchase-order-model-v1 → your trained model — used in Lab 1.3
#
# WHERE THIS IS USED NEXT
# Day 1 Lab 1.3: run after training completes to confirm your custom model ID
# Day 4: reference this output when registering agent tools with model IDs
#
# COMMON ERRORS AND FIXES
# ValueError (missing endpoint/key)  → check your .env file has both values
# HttpResponseError 401              → your API key is wrong, re-copy from portal
# HttpResponseError 404              → your endpoint URL has a typo
# Custom model not appearing         → training is still running, wait and retry
# ModuleNotFoundError                → run: pip install azure-ai-documentintelligence
# =============================================================================
