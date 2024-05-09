import streamlit as st
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

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

openai_api_key = os.getenv("OPENAI_API_KEY")
embedding_function = OpenAIEmbeddings()

def load_data():
    loader = JSONLoader(file_path="./prize3.json", jq_schema=".", text_content=False)
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

def main():
    # # 1. as sidebar menu
    # with st.sidebar:
    #     selected = option_menu("Main Menu", ["Home", 'Settings'], 
    #         icons=['house', 'gear'], menu_icon="cast", default_index=1)
    #     selecteds

    # # 2. horizontal menu
    # selected2 = option_menu(None, ["Home", "Upload", "Tasks", 'Settings'], 
    #     icons=['house', 'cloud-upload', "list-task", 'gear'], 
    #     menu_icon="cast", default_index=0, orientation="horizontal")
    # selected2

    st.title("DineWise AI Menu")

    system_prompt = "You are a world class AI dining companion, so try to be friendly. You are helping a user decide what to eat. You alwyas try to give different options to the user. Always try to give the total cost of the suggested order(that will always be in Rs), whenever you suggest any item. Also always try to suggest something:."
    retriever = load_data()
    chain = setup_chain(retriever, system_prompt)

    # Add buttons for predefined options
    col1, col2= st.columns(2)
    with col1:
        veg_option = st.checkbox("Vegetarian")
        spicy_option = st.checkbox("Spicy")
        under_100 = st.checkbox("Under ₹100")
        breakfast_option = st.checkbox("Breakfast")
        dinner_option = st.checkbox("Dinner")
        combos_option = st.checkbox("Combos")
    with col2:
        non_veg_option = st.checkbox("Non-Vegetarian")
        sweet_option = st.checkbox("Sweet")
        under_500 = st.checkbox("Under ₹500")
        lunch_option = st.checkbox("Lunch")
        snacks_option = st.checkbox("Snacks")
        chinese_option = st.checkbox("Chinese")
        
    prev_qry = ""
    query = st.text_input("What do you want to eat(Anything Specific On Your Mind):")

    options = []
    if veg_option:
        options.append("Vegetarian")
    if non_veg_option:
        options.append("Non-Vegetarian")
    if spicy_option:
        options.append("Spicy")
    if sweet_option:
        options.append("Sweet")
    if under_100:
        options.append("Under ₹100")
    if under_500:
        options.append("Under ₹500")
    if breakfast_option:        
        options.append("Breakfast")
    if lunch_option:
        options.append("Lunch")
    if dinner_option:
        options.append("Dinner")
    if snacks_option:
        options.append("Snacks")
    if combos_option:
        options.append("Combos")
    if chinese_option:
        options.append("Chinese")

    if options:
        query += " (" + ", ".join(options) + ")"

    if st.button("Select For Me") or (prev_qry != query):
        prev_qry = query
        result = chain.invoke(query)
        st.write("Answer:", result)

if __name__ == "__main__":
    main()