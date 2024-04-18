import os
from dotenv import load_dotenv

from langchain import PromptTemplate
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationSummaryBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type
from bs4 import BeautifulSoup
import requests
from langchain.schema import SystemMessage
# from fastapi import FastAPI
import shutil
import streamlit as st

# Libraries to fetch external information
from bs4 import BeautifulSoup
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
import os

load_dotenv()

# 1. Tool for search


options = webdriver.ChromeOptions()
options.add_argument("--disable-notifications")
driver_path = ChromeDriverManager().install()
ChromeService = Service(executable_path=driver_path)
driver = webdriver.Chrome(service=ChromeService, options=options)
ln_login = False


def search(query):
    query_quote = requests.utils.quote(query)
    url = f"https://duckduckgo.com/?t=h_&q={query_quote}"
    driver = webdriver.Chrome(service=ChromeService, options=options)
    driver.get(url)
    driver.implicitly_wait(5)
    # Get the results within the following class and tag
    # <ol class="react-results--main">
    source = driver.page_source
    soup = BeautifulSoup(source, "html.parser")
    
    # Extract the list of elements matching the provided li tag
    matching_elements = soup.find_all('li', 
                                    attrs={'data-layout': 'organic'})
    # Initialize the result dictionary
    result_dict = {
        "query": query,  # This is just an example, you might need to replace it with the actual query.
        "organic_results": []
    }
    for result in matching_elements:
    # Extracting the list of organic results
        for article in result.find_all('article', attrs={'data-nrn': 'result'}):
            
            # Extract position (assuming the ID format is 'r1-<position>')
            position = int(article['id'].split('-')[-1])
            
            # Extract title and link
            title_elem = article.find('a', attrs={'data-testid': 'result-title-a'})
            title = title_elem.text if title_elem else ""
            link = title_elem['href'] if title_elem else ""
            
            # Extract displayed link
            displayed_link_elem = article.find('a', attrs={'data-testid': 'result-extras-url-link'})
            displayed_link = "".join([span.text for span in displayed_link_elem.find_all('span')]) if displayed_link_elem else ""
            
            # Extract snippet
            snippet_elem = article.find('div', attrs={'data-result': 'snippet'})
            snippet = " ".join([span.text for span in snippet_elem.find_all('span') if not span.has_attr('class')]) if snippet_elem else ""
            
            # Add to the results list
            result_dict['organic_results'].append({
                "position": position,
                "title": title,
                "link": link,
                "displayed_link": displayed_link,
                # only display the first 150 characters of the snippet if the snippet is too long
                # "snippet": snippet
                "snippet": snippet[:150] + "..."
            })

    print(result_dict)
    driver.quit()
    return str(result_dict)







# 2. Tool for scraping
def scrape_website(objective: str, url: str):
    # scrape website, and also will summarize the content based on objective if the content is too large
    # objective is the original objective & task that user give to the agent, url is the url of the website to be scraped

    print(f"Scraping website: {url}...")
    driver = webdriver.Chrome()
    driver.get(url)
    driver.implicitly_wait(5)
    content = driver.page_source

    soup = BeautifulSoup(content, "html.parser")
    text = soup.get_text()
    print("CONTENTTTTTT:", text)

    if len(text) > 10000:
        output = summary(objective, text)
        return output
    else:
        return text

def summary(objective, content):
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"], chunk_size=10000, chunk_overlap=500)
    docs = text_splitter.create_documents([content])
    map_prompt = """
    Write a summary of the following text for {objective}:
    "{text}"
    SUMMARY:
    """
    map_prompt_template = PromptTemplate(
        template=map_prompt, input_variables=["text", "objective"])

    summary_chain = load_summarize_chain(
        llm=llm,
        chain_type='map_reduce',
        map_prompt=map_prompt_template,
        combine_prompt=map_prompt_template,
        verbose=True
    )

    output = summary_chain.run(input_documents=docs, objective=objective)

    return output


class ScrapeWebsiteInput(BaseModel):
    """Inputs for scrape_website"""
    objective: str = Field(
        description="The objective & task that users give to the agent")
    url: str = Field(description="The url of the website to be scraped")


class ScrapeWebsiteTool(BaseTool):
    name = "scrape_website"
    description = "useful when you need to get data from a website url, passing both url and objective to the function; DO NOT make up any url, the url should only be from the search results"
    args_schema: Type[BaseModel] = ScrapeWebsiteInput

    def _run(self, objective: str, url: str):
        return scrape_website(objective, url)

    def _arun(self, url: str):
        raise NotImplementedError("error here")


# 3. Create langchain agent with the tools above
tools = [
    Tool(
        name="Search",
        func=search,
        description="useful for when you need to answer questions about current events, data. You should ask targeted questions"
    ),
    ScrapeWebsiteTool(),
]

system_message = SystemMessage(
content="""
### Objective:
You are a world class recruiter, who can do detailed research on any person and make an assessment on if they are suitable for the role based on results.
### Prime Directive:
You do not make things up, you will try as hard as possible to gather facts & data to back up the research.
### Rules:
1. You should do enough research to gather as much information as possible about the objective.
2. If there are url of relevant links & articles, you will scrape it to gather more information.
3. After scraping & search, you should think "is there any new things I should search & scraping based on the data I collected to increase research quality?" If answer is yes, continue; But don't do this more than 10 iteratins
4. You should not make things up, you should only write facts & data that you have gathered.
5. In the final output, You should include all reference data & links to back up your research; You should include all reference data & links to back up your research.
6. Respond using the following format: 
Reasoning: {reasoning}, Conclusion: {conclusion}, Reference: {reference}"""
)

agent_kwargs = {
    "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
    "system_message": system_message,
}

llm = ChatOpenAI(temperature=0, model="gpt-4-turbo")
memory = ConversationSummaryBufferMemory(
    memory_key="memory", return_messages=True, llm=llm, max_token_limit=1000)

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    agent_kwargs=agent_kwargs,
    memory=memory,
)

# 4. Use streamlit to create a web app
def main():
    st.set_page_config(page_title="Who should I research?", page_icon=":bird:")

    st.header("Who should I research?  :bird:")
    query = st.text_input("Research goal/target")

    if query:
        st.write("Doing research for ", query)

        result = agent({"input": query})

        st.info(result['output'])


if __name__ == '__main__':
    main()

