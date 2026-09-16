import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from pypdf import PdfReader

load_dotenv()

LLM = ChatOpenAI(model_name="gpt-5-nano", temperature=0)



def get_policy_docs():
    """"
    Function to get the policy documents from the specified directory.
    Returns a list of file paths for the policy documents.
    """
    pdf_path = "../Docs/Resume_Python.pdf"
    reader = PdfReader(pdf_path)

    # Loop through pages and construct standard LangChain Document objects
    docs = []
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        
        # Create the Document object matching LangChain's exact internal schema
        doc = Document(
            page_content=text,
            metadata={
                "source": pdf_path,
                "page": page_num
            }
        )
        docs.append(doc)

    # Display the final list of Document objects
    return docs


def split_docs_to_chunks(docs, chunk_size=500, chunk_overlap=25):
    """
    Function to split the documents into chunks of specified size.
    Returns a list of document chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(docs)
    return chunks

def create_and_store_embeddings(chunks, context="get candidate details"):
    """
    Function to create embeddings for the document chunks and store them in a vector store.
    Returns the vector store.
    """
    embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small",
    # openai_api_key=os.getenv("OPENAI_API_KEY")  # Optional: Or use the OPENAI_API_KEY environment variable
    )
    vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    # persist_directory="chroma_db"
    )
    context = vectorstore.similarity_search(context, k=10)
    return context


def invoke_llm_with_context(query, context):
    """
    Function to invoke the LLM with the given query and context.
    Returns the response from the LLM.
    """
    response = LLM.invoke(query + str(context))
    return response.content

def rag_flow(query):
    """
    Function to perform the RAG flow.
    Takes a query and a list of document chunks as input.
    Returns the response from the LLM.
    """
    in_context = "candidate all worked companies"
    docs = get_policy_docs()
    chunks = split_docs_to_chunks(docs)
    context = create_and_store_embeddings(chunks, context=in_context)
    response = invoke_llm_with_context(query, context)
    return response


if __name__ == "__main__":
    query = "How many companies candidate worked in total with time period?"
    # docs = get_policy_docs()
    response = rag_flow(query)
    print(response)