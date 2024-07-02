# Description


This is an example of running an ai app locally using the langchain library.
It is shared with Pinnacle Solutions Group to spread learning of the ai tools and coding.

The design of the project is to retrieve a docs page on the deltrs for python project,
[project docs](https://delta-io.github.io/delta-rs/).  The code uses a prompt to ask the llama3
model to return a summary of the project.

Useful to building this project was the youtube hello world tutorial playlist: [Tutorial - Langchain](https://www.youtube.com/playlist?list=PL4HikwTaYE0GEs7lvlYJQcvKhq0QZGRVn).
The creator walks through his code examples to show creating a basic llm call, then a RAG, and finally
an agent with history.

## Code - Retrieval Augmented Generation

The code is a version of a Retrieval Augmented Generation (RAG).  The ideal is
to feed the llm with documents (corpus of knowledge) that is closer to the questions
in the prompt.  Because llm's are built from data that can be a few years old, it may
not contain the knowledge you are searching for.  A RAG augments the llm's corpus
of knowledge.

The code is a simple version of a RAG to show the parts of langchain that lead to a good
response from the llm.

The steps start with docs being pulled from a webpage.
The docs are then split into chunks for embedding creation.  The embeddings are created by the llm.
They are a vector of integers based on the words in the docs.  The reason for chunking is
to keep the size down for llm embedding processing.  Additional to chunk size is the overlap.
The overlap is to help with not splitting context of documents.

After the embeddings are created they are stored in a vector database which is then used
to pull the relevant embeddings later when the chain is devised.

The chain is built using a prompt, a retriever, and an llm.  The retriever comes from
the vector database and langchain converts it to an object that is useful for the llm.
From the chain, the llm produces the response.

[langchain - RAG](https://python.langchain.com/v0.2/docs/tutorials/rag/)

## Running

The simple instructions are:
- set up python virtualenv through poetry
- get ollama running in a terminal: [Ollama Setup](#ollama-setup)
- in new terminal run python code: `PYTHONPATH=. poetry run python src/code.py`

The code will pull the webpages, create the embeddings, store in vectore store,
then call llama3 with the prompt and retriever.  The answer will be printed to stdout.

## Ollama Setup

Ollama is tool that runs locally. You can pull different llm models.  [Ollama Site](https://ollama.com/)

- Install Ollama: `brew install ollama`
- Start ollam locally: `ollama serve`
- Retrieve the llama3 model: `ollama pull llama3`

## Langsmith

LangSmith is a browser based tool that displays tracing of the run of a model and
adds the ability to do testing of your models.

Instructions for using LangSmith is on the [LangSmith webpage](https://smith.langchain.com/).
The basic setup is to add `LANGCHAIN_TRACING_V2` and `LANGCHAIN_API_KEY`
as env vars.  LangChain will then make the calls to LangSmith when
your code runs.  I haven't dug in too much on making use of LangSmith
but the visual of the parts of your model does help when learning LangChain.

## Note about Agents and Chat History

LangChain has the ability to run agents.  The short answer of what an agent is, they are
part of an llm that can make decisions on how to process inputs, as well as contain history
of the prompts from the user.  As an example, if you ask if LangChain is good for testing
model setup, you can then ask the subsequent quetion of show me how.

[Agent vs Chain](https://www.restack.io/docs/langchain-knowledge-langchain-agents-vs-chains)

## Answers Directory

There is an `answers` directory which is a collection of the answers given when the chunk size changes.
The number appendended to the filename is the chunksize.

Of interest is how the answer changes based on the chunk size.
