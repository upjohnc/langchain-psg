from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS, VectorStore
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable


def get_docs(chunk_size: int) -> list[Document]:
    loader = WebBaseLoader("https://delta-io.github.io/delta-rs/")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=20,
    )
    split_docs = text_splitter.split_documents(docs)

    return split_docs


def create_vector_store(docs: list[Document]) -> VectorStore:
    embeddings = OllamaEmbeddings()
    vector_store = FAISS.from_documents(docs, embeddings)
    return vector_store


def create_chain(vector_store: VectorStore) -> Runnable:
    model = Ollama(model="llama3")
    prompt = ChatPromptTemplate.from_template(
        """
            Answer the user's question.
            Context: {context}
            Question: {input}
            """
    )
    document_chain = create_stuff_documents_chain(llm=model, prompt=prompt)
    retriever = vector_store.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    return retrieval_chain


def main():
    chunk_size = 500
    docs = get_docs(chunk_size)
    vector_store = create_vector_store(docs)
    chain = create_chain(vector_store)
    response = chain.invoke(dict(input="Summarize what deltalake is"))
    print("Here is your answer:")
    print(response["answer"])

    with open("response_capture", "w") as f:
        f.write(str(response))
    return response


if __name__ == "__main__":
    _ = main()
