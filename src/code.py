from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.retrievers import ParentDocumentRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS, Chroma, VectorStore
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.stores import InMemoryStore


def get_docs(chunk_size: int) -> tuple[list[Document], RecursiveCharacterTextSplitter]:
    loader = WebBaseLoader("https://delta-io.github.io/delta-rs/")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=20,
    )
    return docs, text_splitter


def get_split_docs(chunk_size: int) -> list[Document]:
    docs, text_splitter = get_docs(chunk_size)
    split_docs = text_splitter.split_documents(docs)

    return split_docs


def create_vector_store(docs: list[Document]) -> VectorStore:
    embeddings = OllamaEmbeddings(model="llama3")
    vector_store = FAISS.from_documents(docs, embeddings)
    return vector_store


def create_parent_doc_retriever(chunk_size: int):
    """
    Create a retriever from a parent document retriever

    The retriever fetches the small chunks of documents and then ties
    those chunks to the parent document and retrieves the parent document.
    https://python.langchain.com/v0.1/docs/modules/data_connection/retrievers/parent_document_retriever/

    Args:
        chunk_size (int): the size of the chunks from the text splitting
    """
    docs, text_splitter = get_docs(chunk_size)
    embeddings = OllamaEmbeddings(model="llama3")
    vector_store = Chroma(
        collection_name="full_documents", embedding_function=embeddings
    )
    docstore = InMemoryStore()
    retriever = ParentDocumentRetriever(
        vectorstore=vector_store, docstore=docstore, child_splitter=text_splitter
    )
    retriever.add_documents(docs, ids=None)
    return retriever


def create_chain(retriever) -> Runnable:
    model = Ollama(model="llama3")
    prompt = ChatPromptTemplate.from_template(
        """
        Answer the user's question.
        Context: {context}
        Question: {input}
        """
    )
    document_chain = create_stuff_documents_chain(llm=model, prompt=prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    return retrieval_chain


def main():
    run_parent_document_retriever = True
    chunk_size = 500

    if run_parent_document_retriever:
        retriever = create_parent_doc_retriever(chunk_size)
    else:
        split_docs = get_split_docs(chunk_size)
        vector_store = create_vector_store(split_docs)
        retriever = vector_store.as_retriever()

    chain = create_chain(retriever)

    response = chain.invoke(
        dict(input="Summarize what deltalake is")
    )  # note: deltalake is the name of the package

    print("Here is your answer:")
    print(response["answer"])

    with open("response_capture", "w") as f:
        f.write(str(response))

    return response


if __name__ == "__main__":
    _ = main()
