import time

import requests
from bs4 import BeautifulSoup
from langchain.chains import RetrievalQA
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOpenAI

# import openai
# openai.api_key = ""
JINA_LINK = "https://r.jina.ai/"
CRAWL_LINK = "https://www.llamaindex.ai/blog/"


def save_file_local(text, file_name):
    with open(f"{file_name}", "w") as text_file:
        text_file.write(text)


def crawl_data(save_file=False):
    # save_file True run only first time for crawl all data
    response = requests.get(f"{JINA_LINK}{CRAWL_LINK}")
    links = response.text.split("\n")
    exclude_links = ['URL Source: https://www.llamaindex.ai/blog/']
    correct_text_link = []
    for item in links:
        if 'image' in item:
            continue
        if 'https' not in item:
            continue
        if item in exclude_links:
            continue
        try:
            correct_link = item.split("](")[1][:-1]
            correct_text_link.append(correct_link)
            exclude_links.append(correct_link)
        except Exception as e:
            # import pdb
            # pdb.set_trace()
            pass

    if save_file:
        for link in correct_text_link:
            file_name = f'{link.replace(f"{CRAWL_LINK}", "")}.txt'
            print(f"crawl {link}")
            wrong_contents = ["RateLimitTriggeredError", "requests per minute per IP address", "A timeout occurred"]
            try:
                with open(file_name, 'r') as file:
                    contents = file.read()
            except:
                contents = "RateLimitTriggeredError"
            for item in wrong_contents:
                if item in contents:
                    response = requests.get(f"{JINA_LINK}{link}")
                    while 'hit the rate limit (requests per minute per IP address).' in response.text or 'RateLimitTriggeredError' in response.text:
                        time.sleep(60)
                        response = requests.get(f"{JINA_LINK}{link}")

                    save_file_local(response.text, file_name)

    all_files = []
    for link in correct_text_link:
        file_name = f'{link.replace(f"{CRAWL_LINK}", "")}.txt'
        all_files.append(file_name)
    return all_files


if __name__ == '__main__':
    all_files = crawl_data(save_file=False)
    # print(all_files)
    all_documents = []
    for file_name in all_files:
        with open(file_name, 'r') as file:
            html_content = file.read()
        contents = html_content.split("\n\n")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        page_metadata = {}
        all_content = []
        for content_item in contents:
            if 'Title: ' in content_item:
                page_metadata.update({
                    'title': content_item.split('Title: ')[1]
                })
            elif 'URL Source: ' in content_item:
                page_metadata.update({
                    'source': content_item.split('URL Source: ')[1]
                })
            else:
                all_content.append(content_item)

        documents = text_splitter.create_documents(all_content, metadatas=[page_metadata])
        all_documents.extend(documents)

    # Initialize embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3, openai_api_key=openai_api_key)

    vectorstore = FAISS.from_documents(all_documents, embeddings)

    system_prompt = (
        "Given the following context, provide a concise answer to the question."
        "\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(vectorstore.as_retriever(), question_answer_chain)

    query_1 = 'What are key features of llama-agents?'
    # query_2 = 'What are the two critical areas of RAG system performance that are assessed in the "Evaluating RAG with LlamaIndex" section of the OpenAI Cookbook?'
    # query_3 = 'What are the two main metrics used to evaluate the performance of the different rerankers in the RAG system?'
    for query in [query_1]:
        response = rag_chain.invoke({"input": f"{query}"})
        print(query)
        print("==============")
        print(response)
