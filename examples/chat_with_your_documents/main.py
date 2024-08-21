import argparse
import os
from typing import List
from openai.types.chat import ChatCompletionMessageParam
import openai
import chromadb


def build_prompt(query: str, context: List[str]) -> List[ChatCompletionMessageParam]:
    """
    Builds a prompt for the LLM. #

    This function builds a prompt for the LLM. It takes the original query,
    and the returned context, and asks the model to answer the question based only
    on what's in the context, not what's in its weights.

    More information: https://platform.openai.com/docs/guides/chat/introduction

    Args:
    query (str): The original query.
    context (List[str]): The context of the query, returned by embedding search.

    Returns:
    A prompt for the LLM (List[ChatCompletionMessageParam]).
    """

    system: ChatCompletionMessageParam = {
        "role": "system",
        "content": "I am going to ask you a question, which I would like you to answer"
        "based only on the provided context, and not any other information."
        "If there is not enough information in the context to answer the question,"
        'say "I am not sure", then try to make a guess.'
        "Break your answer up into nicely readable paragraphs.",
    }
    user: ChatCompletionMessageParam = {
        "role": "user",
        "content": f"The question is {query}. Here is all the context you have:"
        f'{(" ").join(context)}',
    }

    return [system, user]


def get_chatGPT_response(query: str, context: List[str], model_name: str) -> str:
    """
    Queries the GPT API to get a response to the question.

    Args:
    query (str): The original query.
    context (List[str]): The context of the query, returned by embedding search.

    Returns:
    A response to the question.
    """
    response = openai.chat.completions.create(
        model=model_name,
        messages=build_prompt(query, context),
    )

    return response.choices[0].message.content  # type: ignore


import os
import requests
import json
import argparse
import chromadb

def get_azure_openai_response(prompt: str, model_name: str) -> str:
    # Retrieve Azure OpenAI API key and endpoint from environment variables
    api_key = os.getenv("e6e399c281c84e9da226cb96d34c2f3a")
    endpoint = os.getenv("https://madhaviopenai1.openai.azure.com/")
    api_version = "2023-08-01"  # Ensure this matches the version you are using

    if not api_key or not endpoint:
        raise ValueError("Azure OpenAI API key or endpoint not set")

    # Define the headers for authentication
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    # Define the URL for the API request
    url = f"{endpoint}/v1/engines/{model_name}/completions"

    # Define the payload for the API request
    payload = {
        "prompt": prompt,
        "max_tokens": 100,  # Adjust max_tokens as needed
        "temperature": 0.7  # Adjust temperature as needed
    }

    # Make the API request
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()  # Raise an exception for HTTP errors
        result = response.json()
        return result.get("choices", [{}])[0].get("text", "").strip()
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return "Error in response"

def main(
    collection_name: str = "documents_collection", persist_directory: str = "."
) -> None:
    # Check if the Azure OpenAI API key and endpoint environment variables are set.
    if "AZURE_OPENAI_API_KEY" not in os.environ or "AZURE_OPENAI_ENDPOINT" not in os.environ:
        print("Please set AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT environment variables.")
        return

    # Ask what model to use
    model_name = "gpt-3.5-turbo"
    answer = input(f"Do you want to use GPT-4? (y/n) (default is {model_name}): ")
    if answer == "y":
        model_name = "gpt-4"

    # Instantiate a persistent Chroma client in the persist_directory.
    client = chromadb.PersistentClient(path=persist_directory)

    # Get the collection.
    collection = client.get_collection(name=collection_name)

    # Use a simple input loop.
    while True:
        # Get the user's query
        query = input("Query: ")
        if len(query) == 0:
            print("Please enter a question. Ctrl+C to Quit.\n")
            continue
        print(f"\nThinking using {model_name}...\n")

        # Query the collection to get the 5 most relevant results
        results = collection.query(
            query_texts=[query], n_results=5, include=["documents", "metadatas"]
        )

        sources = "\n".join(
            [
                f"{result['filename']}: line {result['line_number']}"
                for result in results["metadatas"][0]  # type: ignore
            ]
        )

        # Get the response from Azure OpenAI
        response = get_azure_openai_response(query, model_name)  # type: ignore

        # Output, with sources
        print(response)
        print("\n")
        print(f"Source documents:\n{sources}")
        print("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load documents from a directory into a Chroma collection"
    )

    parser.add_argument(
        "--persist_directory",
        type=str,
        default="chroma_storage",
        help="The directory where you want to store the Chroma collection",
    )
    parser.add_argument(
        "--collection_name",
        type=str,
        default="documents_collection",
        help="The name of the Chroma collection",
    )

    # Parse arguments
    args = parser.parse_args()

    main(
        collection_name=args.collection_name,
        persist_directory=args.persist_directory,
    )
