import os  # used to access environment variables
from dotenv import load_dotenv  # loads variables from .env file
from azure.search.documents import SearchClient  # client to perform search queries
from azure.search.documents.models import VectorizedQuery  # used for vector-based queries
from azure.core.credentials import AzureKeyCredential  # used for authentication
from langchain_openai import AzureOpenAIEmbeddings  # used to generate embeddings
 
load_dotenv()  # load environment variables from .env file
 
# create a search client to connect to Azure AI Search
search_client = SearchClient(
    endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),  # Azure Search endpoint
    index_name="documents-index",  # name of the index to query
    credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_KEY"))  # API key
)
 
# create embeddings model using Azure OpenAI
embeddings_model = AzureOpenAIEmbeddings(
    azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),  # embedding model name
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),  # Azure OpenAI endpoint
    api_key=os.getenv("AZURE_OPENAI_KEY"),  # API key
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),  # API version
)
 
 
def keyword_search(query, top=3):
    """Standard BM25 keyword search."""
    # perform keyword-based search using text matching
    results = search_client.search(
        search_text=query,  # query text
        top=top,  # number of results to return
        select=["id","content","document_type","source_file","confidence"]  # fields to return
    )
    return list(results)  # convert results to list
 
 
def vector_search(query, top=3):
    """Pure vector similarity search."""
    vector = embeddings_model.embed_query(query)  # convert query into embedding vector
 
    # create vector query using embedding
    vec_query = VectorizedQuery(
        vector=vector,
        k_nearest_neighbors=top,  # number of nearest matches
        fields="content_vector"  # vector field in index
    )
 
    # perform search using only vector similarity
    results = search_client.search(
        search_text=None,  # no keyword search
        vector_queries=[vec_query],
        top=top,
        select=["id","content","document_type","source_file","confidence"]
    )
    return list(results)
 
 
def hybrid_search(query, top=3):
    """Hybrid keyword + vector search with RRF fusion."""
    vector = embeddings_model.embed_query(query)  # convert query into embedding
 
    # create vector query
    vec_query = VectorizedQuery(
        vector=vector,
        k_nearest_neighbors=top,
        fields="content_vector"
    )
 
    # perform hybrid search combining keyword and vector
    results = search_client.search(
        search_text=query,  # keyword part
        vector_queries=[vec_query],  # vector part
        top=top,
        select=["id","content","document_type","source_file","confidence"]
    )
    return list(results)
 
 
def print_results(results, mode):
    # print search results in readable format
    print(f"\n--- {mode} ---")
 
    for i, r in enumerate(results, 1):
        print(f"{i}. [{r['document_type']}] {r['source_file']}")  # document info
        print(f"   Score: {r['@search.score']:.4f}")  # relevance score
        print(f"   Excerpt: {r['content'][:120]}...")  # short preview of content
 
 
if __name__ == "__main__":
    # list of sample queries to test search modes
    queries = [
        "What is the total amount on the Contoso invoice?",
        "patient chest pain medication allergy",
        "VAT payment terms bank details",
    ]
 
    # run all three search types for each query
    for query in queries:
        print(f"\n=== Query: {query} ===")
        print_results(keyword_search(query), "Keyword Search")  # keyword results
        print_results(vector_search(query),  "Vector Search")  # vector results
        print_results(hybrid_search(query),  "Hybrid Search")  # hybrid results