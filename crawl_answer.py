import re
import time

import requests
from bs4 import BeautifulSoup
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tenacity import retry, stop_after_attempt, wait_exponential

OPENAI_API_KEY = ""
JINA_LINK = "https://r.jina.ai/"
CRAWL_LINK = "https://www.llamaindex.ai/blog/"
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}


def save_file_local(text, file_name):
    with open(f"{file_name}", "w") as text_file:
        text_file.write(text)


def get_all_sub_link_of_blog():
    # save_file have to True when run first time for crawl all data to local
    response = requests.get(f"{JINA_LINK}{CRAWL_LINK}")
    links = response.text.split("\n")
    # exclude itself from all sub blog need crawl
    exclude_links = ['URL Source: https://www.llamaindex.ai/blog/']
    correct_text_link = []
    # get all sub link of blog
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
            print(e)
            pass
    return correct_text_link


def crawl_data_by_jina(save_file=False):
    # save_file have to True when run first time for crawl all data to local
    correct_text_link = get_all_sub_link_of_blog()

    if save_file:
        # save jina format crawl data of all sub link, sleep wait 60s if error rate limit
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


# Extracts blog detail content.
def extract_blog_detail_info(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    # Extract the title
    title = soup.find('h1').get_text(strip=True)

    # Extract content from the section with class 'BlogPost_htmlPost'
    content_section = soup.find('div', class_=re.compile(r'BlogPost_htmlPost'))
    content = []

    if len(content_section.find_all(['h1', 'h2', 'h3', 'strong'])) == 0:
        content.append({'header': title, 'content': content_section.get_text(strip=True)})

    for section in content_section.find_all(['h1', 'h2', 'h3', 'strong']):
        header = section.get_text()
        section_content = []
        next_sibling = section.find_next_sibling()
        # Extract section contents
        while next_sibling and next_sibling.name not in {'h1', 'h2', 'h3', 'strong'}:
            section_content.append(next_sibling.get_text(strip=True))
            next_sibling = next_sibling.find_next_sibling()
        content.append({'header': header, 'content': ''.join(section_content)})

    return {
        'title': title,
        'content': content
    }


def create_chunk_document_by_soup(all_files):
    chunks = []
    text_splitter = RecursiveCharacterTextSplitter()
    for file_name in all_files:
        url_suffix = file_name.replace(".txt", "")
        url = f'{CRAWL_LINK}{url_suffix}'
        request_blog = requests.get(url, headers=headers, timeout=10)
        request_blog.raise_for_status()
        blog_detail = extract_blog_detail_info(request_blog.content)
        for blog_section in blog_detail.get('content'):
            if len(blog_section.get('content')) > 0:
                chunks += ['Blog title: ' + blog_detail.get('title') + '\n' + 'Section: ' + blog_section.get(
                    'header') + '\n' + 'Content: ' + blog_section.get('content')]
    return text_splitter.create_documents(chunks)


def create_chunk_use_jina(all_files):
    all_documents = []
    for file_name in all_files:
        with open(file_name, 'r') as file:
            html_content = file.read()
        contents = html_content.split("\n\n")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        page_metadata = {}
        all_content = []
        metadatas = []
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
                metadatas.append(page_metadata)
        try:
            documents = text_splitter.create_documents(all_content, metadatas=metadatas)
            all_documents.extend(documents)
        except Exception as e:
            print(e)
    return all_documents


# Retry configuration
@retry(stop=stop_after_attempt(5), wait=wait_exponential(min=1, max=60))
def safe_invoke(chain, input_data):
    return chain.invoke(input_data)


if __name__ == '__main__':
    # load stored data for saving money
    vectorstore = None
    try:
        vectorstore = FAISS.load_local("faiss_store", OpenAIEmbeddings())
    except:
        pass

    # Initialize embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2, openai_api_key=OPENAI_API_KEY)
    if not vectorstore:
        all_files = crawl_data_by_jina(save_file=False)
        # all_documents = create_chunk_use_jina(all_files) # use save_file=True first time run, for crawl all html to local

        all_documents = create_chunk_document_by_soup(all_files)
        vectorstore = FAISS.from_documents(all_documents, embeddings)

    system_prompt = (
        "Given the following context, provide a concise answer to the question. Answer must have title and source"
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
    rag_chain = create_retrieval_chain(vectorstore.as_retriever(search_kwargs={"k": 2}), question_answer_chain)

    objects_to_save = {
        "question_answer_chain": question_answer_chain,
        "rag_chain": rag_chain,
    }
    query_1 = 'What are key features of llama-agents?'
    query_2 = 'What are the two critical areas of RAG system performance that are assessed in the "Evaluating RAG with LlamaIndex" section of the OpenAI Cookbook?'
    query_3 = 'What are the two main metrics used to evaluate the performance of the different rerankers in the RAG system?'
    for query in [query_1]:
        # response = rag_chain.invoke({"input": f"{query}"})
        # print(query)
        # print("==============")
        # print(response)
        try:
            print(query)
            print("==============")
            response = safe_invoke(rag_chain, {"input": query})
            print(response)
        except Exception as e:
            print(f"Failed after retries: {e}")
