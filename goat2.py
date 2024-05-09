from flask import Flask, request
import requests
from twilio.twiml.messaging_response import MessagingResponse
from openai import OpenAI
app = Flask(__name__)

from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import JSONLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import os
from dotenv import load_dotenv
import langsmith

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")

# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

openai_api_key = os.getenv("OPENAI_API_KEY")
embedding_function = OpenAIEmbeddings()

def load_data():
    loader = JSONLoader(file_path="./prize.json", jq_schema=".", text_content=False)
    documents = loader.load()
    db = Chroma.from_documents(documents, embedding_function)
    retriever = db.as_retriever()
    return retriever

def setup_chain(retriever, system_prompt):
    template = """Answer the question based only on the following context in a conversational tone. Never start with based on the context. You are a world class AI dining companion, so try to be friendly. Remember the Non-Veg Options are usually with chicken. Here are the broad categories/headings for typical food eaten at Bikanervala restaurants across different meal times:
    Breakfast:
    Snacks/Chaat
    Thali/Combos
    Bread/Bakery Items
    Lunch:
    Thali Meals
    Curries/Gravies
    Breads/Rice
    Evening:
    Chaat/Snacks
    Sweets
    Dinner:
    Vegetarian Main Course
    Dal/Sabzi Curries
    Breads/Rice: {context} Question: {question} """
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_prompt),
        HumanMessagePromptTemplate.from_template(template)
    ])
    model = ChatOpenAI()
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    return chain

@app.route('/bot', methods=['POST'])
def bot():
    system_prompt = "You are a world class AI dining companion, so try to be friendly. You are helping a user decide what to eat. You alwyas try to give different options to the user. Always try to give the total cost of the suggested order(that will always be in Rs), whenever you suggest any item. Also always try to suggest something:."
    retriever = load_data()
    chain = setup_chain(retriever, system_prompt)
    
    incoming_msg = request.values.get('Body', '').lower()
    resp = MessagingResponse()
    msg = resp.message()
    responded = False
    
    client = OpenAI()
    response = client.chat.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": incoming_msg}
    ]
)
    print(response)
    print(response.choices[0].text)
    
    # if True: 
    msg.body(response.choices[0].text)
    responded = True
        # # return a quote
        # r = requests.get('https://api.quotable.io/random')
        # if r.status_code == 200:
        #     data = r.json()
        #     quote = f'{data["content"]} ({data["author"]})'
        # else:
        #     quote = 'I could not retrieve a quote at this time, sorry.'
        # msg.body(quote)
    # if 'cat' in incoming_msg:
    #     # return a cat pic
    #     msg.media('https://cataas.com/cat')
    #     responded = True
    if not responded:
        msg.body('I only know about famous quotes and cats, sorry!')
    return str(resp)


if __name__ == '__main__':
    app.run(port=4000)