import os  # used to access environment variables
import uuid  # used to generate unique IDs for each document
import json  # used for working with JSON data
from dotenv import load_dotenv  # loads environment variables from .env file
from azure.search.documents import SearchClient  # client to upload and search documents
from azure.core.credentials import AzureKeyCredential  # used for authentication
from langchain_openai import AzureOpenAIEmbeddings  # used to generate embeddings
 
load_dotenv()  # load variables from .env file into environment
 
# read Azure Search configuration from environment variables
SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
SEARCH_KEY      = os.getenv("AZURE_SEARCH_KEY")
INDEX_NAME      = "documents-index"  # name of the search index
 
# create a client to interact with Azure AI Search index
search_client = SearchClient(
    endpoint=SEARCH_ENDPOINT,
    index_name=INDEX_NAME,
    credential=AzureKeyCredential(SEARCH_KEY)
)
 
# create embeddings model using Azure OpenAI
embeddings_model = AzureOpenAIEmbeddings(
    azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),  # embedding model deployment name
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),  # Azure OpenAI endpoint
    api_key=os.getenv("AZURE_OPENAI_KEY"),  # API key for authentication
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),  # API version
)
 
 
def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into overlapping chunks by word count."""
    words = text.split()  # split text into individual words
    chunks = []  # list to store chunks
    start = 0  # starting index
 
    # loop until all words are processed
    while start < len(words):
        end = min(start + chunk_size, len(words))  # define end of chunk
        chunks.append(" ".join(words[start:end]))  # create chunk and add to list
        start += chunk_size - overlap  # move forward with overlap
 
    return chunks  # return all chunks
 
 
def index_document(content, metadata):
    """
    Chunk a document, generate embeddings, and upload to Azure AI Search.
    metadata contains details like document type, file name, vendor, etc.
    """
    chunks = chunk_text(content)  # split content into chunks
    print(f"  Chunked into {len(chunks)} segments")  # print number of chunks
 
    documents = []  # list to store documents for upload
 
    # loop through each chunk
    for i, chunk in enumerate(chunks):
        vector = embeddings_model.embed_query(chunk)  # generate embedding for chunk
 
        # create document object
        doc = {
            "id":             str(uuid.uuid4()),  # unique ID
            "content":        chunk,  # actual text chunk
            "content_vector": vector,  # embedding vector
            "chunk_index":    i,  # position of chunk
            "document_type":  metadata.get("document_type", "unknown"),  # type of document
            "source_file":    metadata.get("source_file", ""),  # file name
            "vendor_name":    metadata.get("vendor_name", ""),  # vendor name
            "invoice_date":   metadata.get("invoice_date", ""),  # invoice date
            "confidence":     metadata.get("confidence", 0.0),  # extraction confidence
            "page_number":    metadata.get("page_number", 1),  # page number
        }
 
        documents.append(doc)  # add document to list
 
    # upload documents in batches of 100
    batch_size = 100
 
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]  # create batch
        result = search_client.upload_documents(documents=batch)  # upload batch
 
        # count how many documents succeeded
        succeeded = sum(1 for r in result if r.succeeded)
        print(f"  Uploaded batch {i//batch_size + 1}: {succeeded}/{len(batch)} documents indexed")
 
    return len(documents)  # return total number of indexed chunks
 
 
if __name__ == "__main__":
    # sample documents similar to output from earlier processing scripts
    sample_docs = [
        {
            "content": """Contoso Consulting Invoice INV-2024-0892.
            Vendor: Contoso Consulting Ltd, 123 Business Park, London EC1A 1BB.
            Customer: Acme Corporation, 456 Corporate Drive, Manchester M1 2AB.
            Invoice Date: 15 November 2024. Due Date: 15 December 2024.
            Services rendered: Strategic consulting services Q4 2024.
            Subtotal: GBP 120,000.00. VAT 20%: GBP 24,000.00.
            Total Due: GBP 144,000.00. Payment Terms: Net 30 days.
            Bank: Barclays. Sort Code: 20-00-00. Account: 12345678.""",
            "metadata": {
                "document_type": "invoice",
                "source_file": "INV-2024-0892.pdf",
                "vendor_name": "Contoso Consulting",
                "invoice_date": "2024-11-15",
                "confidence": 0.94,
                "page_number": 1
            }
        },
        {
            "content": """City General Hospital Patient Intake Form.
            Patient Name: Sarah Johnson. Date of Birth: 14 March 1982.
            Insurance Provider: Blue Shield. Member ID: BS-2024-778821.
            Primary Physician: Dr. Amir Patel. Referral Date: 20 November 2024.
            Chief Complaint: Persistent chest pain and shortness of breath for 3 days.
            Allergies: Penicillin. Current Medications: Metformin 500mg twice daily.
            Emergency Contact: Michael Johnson (husband) 07700 900123.""",
            "metadata": {
                "document_type": "intake_form",
                "source_file": "intake-sarah-johnson.pdf",
                "vendor_name": "",
                "invoice_date": "2024-11-20",
                "confidence": 0.91,
                "page_number": 1
            }
        },
    ]
 
    print("=== Document Indexer ===")  # start message
    total = 0  # track total indexed documents
 
    # loop through each sample document
    for doc in sample_docs:
        print(f"\nIndexing: {doc['metadata']['source_file']}")  # print file name
        count = index_document(doc["content"], doc["metadata"])  # index document
        total += count  # update total count
        print(f"  Total chunks indexed: {count}")  # print chunk count
 
    print(f"\nIndexing complete. Total documents in index: {total}")  # final message