import os
from dotenv import load_dotenv
from azure.ai.documentintelligence import DocumentIntelligenceAdministrationClient
from azure.ai.documentintelligence.models import ComposeDocumentModelRequest, ComponentDocumentModelDetails
from azure.core.credentials import AzureKeyCredential

load_dotenv()

admin_client = DocumentIntelligenceAdministrationClient(
    endpoint=os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT"),
    credential=AzureKeyCredential(os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY"))
)

request = ComposeDocumentModelRequest(
    model_id="document-processing-v1",
    description="Composed model for document processing pipeline",
    components=[
        ComponentDocumentModelDetails(model_id="purchase-order-model-v1")
    ]
)

poller = admin_client.begin_compose_model(request)
result = poller.result()

print(f"Composed model created : {result.model_id}")
print(f"Description            : {result.description}")
print(f"Component models       : {[c.model_id for c in result.components]}")
print("\nThis is the endpoint your LangGraph agent will call on Day 4.")
