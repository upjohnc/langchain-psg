# Description

This is an example of running an ai app locally using the langchain library.
It is shared with Pinnacle Solutions Group to spread learning of the ai tools and coding.

The design of the project is to retrieve a docs page on the deltrs for python project,
[project docs](https://delta-io.github.io/delta-rs/).  The code uses a prompt to ask the llama3
model to return a summary of the project.

Useful to building this project was the youtube hello world tutorial playlist: [Tutorial - Langchain](https://www.youtube.com/playlist?list=PL4HikwTaYE0GEs7lvlYJQcvKhq0QZGRVn)
The creator walks through his code examples to show creating a basic llm call, then a RAG, and finally
an agent with history.

## Running

- ollama running

## Ollama Setup

- brew install ollama
- ollama serve
- ollama pull llama3

## Langsmith

https://smith.langchain.com/
