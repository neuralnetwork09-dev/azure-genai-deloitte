import os
from dotenv import load_dotenv
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
from azure.core.credentials import AzureKeyCredential

load_dotenv()

endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
key = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")

if not endpoint:
    raise ValueError("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT is missing from your .env file")
if not key:
    raise ValueError("AZURE_DOCUMENT_INTELLIGENCE_KEY is missing from your .env file")

print(f"Endpoint loaded: {endpoint}")

client = DocumentIntelligenceClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(key)
)

sample_url = (
    "https://raw.githubusercontent.com/Azure-Samples/"
    "cognitive-services-REST-api-samples/master/curl/"
    "form-recognizer/sample-invoice.pdf"
)

print("Sending document to Azure AI Document Intelligence...")

poller = client.begin_analyze_document(
    "prebuilt-read",
    AnalyzeDocumentRequest(url_source=sample_url)
)

result = poller.result()

print("\nConnection successful!")
print(f"Pages analysed : {len(result.pages)}")

for page in result.pages:
    print(f"\nPage {page.page_number}:")
    print(f"  Dimensions : {page.width:.1f} x {page.height:.1f} {page.unit}")
    print(f"  Words found: {len(page.words)}")
    print("  First five words:")
    for word in page.words[:5]:
        print(f"    '{word.content}' (confidence: {word.confidence:.0%})")

print("\nSetup complete. You are ready for Lab 1.2.")
