
import os  # used to access environment variables from your system
from dotenv import load_dotenv  # loads variables from a .env file into environment
from azure.search.documents.indexes import SearchIndexClient  # client to manage indexes in Azure AI Search
from azure.search.documents.indexes.models import (
    SearchIndex, SimpleField, SearchableField, SearchField,  # classes to define index structure
    SearchFieldDataType, VectorSearch,  # data types and vector search config
    HnswAlgorithmConfiguration, VectorSearchProfile,  # vector search algorithm settings
    SemanticConfiguration, SemanticSearch, SemanticPrioritizedFields,  # semantic search config
    SemanticField
)
from azure.core.credentials import AzureKeyCredential  # used to authenticate with Azure using API key
 
load_dotenv()  # load environment variables from .env file
 
# read Azure Search service details from environment variables
SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
SEARCH_KEY      = os.getenv("AZURE_SEARCH_KEY")
INDEX_NAME      = "documents-index"  # name of the index to create
 
# create a client to interact with Azure AI Search index service
index_client = SearchIndexClient(
    endpoint=SEARCH_ENDPOINT,
    credential=AzureKeyCredential(SEARCH_KEY)
)
 
def create_search_index():
    """Create the Azure AI Search index with vector and semantic search support."""
 
    # define vector search configuration using HNSW algorithm
    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(name="hnsw-config")  # define algorithm config name
        ],
        profiles=[
            VectorSearchProfile(
                name="hnsw-profile",  # profile name used in vector field
                algorithm_configuration_name="hnsw-config"  # link to algorithm config
            )
        ]
    )
 
    # define semantic search configuration
    semantic_config = SemanticConfiguration(
        name="semantic-config",  # name of semantic configuration
        prioritized_fields=SemanticPrioritizedFields(
            content_fields=[SemanticField(field_name="content")]  # main text field for semantic ranking
        )
    )
    semantic_search = SemanticSearch(configurations=[semantic_config])  # attach semantic config
 
    # define the structure of the index (fields and their properties)
    fields = [
        SimpleField(
            name="id",
            type=SearchFieldDataType.String,
            key=True,  # unique identifier for each document
            filterable=True
        ),
        SearchableField(
            name="content",
            type=SearchFieldDataType.String,
            analyzer_name="en.microsoft"  # enables full-text search with English language processing
        ),
        SimpleField(
            name="document_type",
            type=SearchFieldDataType.String,
            filterable=True,
            facetable=True  # allows grouping/filtering in UI
        ),
        SimpleField(
            name="source_file",
            type=SearchFieldDataType.String,
            filterable=True
        ),
        SimpleField(
            name="page_number",
            type=SearchFieldDataType.Int32,
            filterable=True,
            sortable=True
        ),
        SimpleField(
            name="chunk_index",
            type=SearchFieldDataType.Int32,
            filterable=True
        ),
        SimpleField(
            name="vendor_name",
            type=SearchFieldDataType.String,
            filterable=True,
            facetable=True
        ),
        SimpleField(
            name="invoice_date",
            type=SearchFieldDataType.String,
            filterable=True
        ),
        SimpleField(
            name="confidence",
            type=SearchFieldDataType.Double,
            filterable=True,
            sortable=True
        ),
        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),  # array of float numbers
            searchable=True,
            vector_search_dimensions=1536,  # size of embedding vector (must match model output)
            vector_search_profile_name="hnsw-profile"  # link to vector search profile
        ),
    ]
 
    # create the index object with all configurations
    index = SearchIndex(
        name=INDEX_NAME,
        fields=fields,
        vector_search=vector_search,
        semantic_search=semantic_search
    )
 
    try:
        # create or update the index in Azure
        result = index_client.create_or_update_index(index)
        print(f"Index created/updated: {result.name}")  # print index name
        print(f"Fields: {[f.name for f in result.fields]}")  # print all field names
    except Exception as e:
        print(f"Error creating index: {e}")  # print error if something fails
        raise  # re-throw error for debugging
 
# run the script only when executed directly (not when imported)
if __name__ == "__main__":
    print("Creating Azure AI Search index...")  # start message
    create_search_index()  # call function to create index
    print("Done. Index is ready for document ingestion.")  # completion message

