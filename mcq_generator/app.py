import streamlit as st
from pdfminer.high_level import extract_text
from langchain.schema import HumanMessage
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_openai import ChatOpenAI
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.text_splitter import RecursiveCharacterTextSplitter
import string
import re

def postprocess_text(text):
    printable = (
        string.ascii_letters +
        string.digits +
        string.punctuation +
        string.whitespace +
        'АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ' +
        'абвгдеёжзийклмнопрстуфхцчшщъыьэюя'
    )
    cleaned_text = ''.join(char for char in text if char in printable)
    cleaned_text = re.sub('\xad(\x0c)*', '', cleaned_text)
    cleaned_text = re.sub('[\n-\x0c]', '', cleaned_text)
    return cleaned_text

st.title("Генератор вопросов")

uploaded_file = st.file_uploader("Загрузите PDF-файл", type="pdf")

if uploaded_file is not None:

    full_text = extract_text(uploaded_file)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_text(full_text)

    chunk_index = st.selectbox("Выберите чанк", range(len(chunks)))
    selected_chunk = postprocess_text(chunks[chunk_index])

    st.write("Выбранный чанк:")
    st.write(selected_chunk)

    if st.button("Сгенерировать вопросы по чанку"):

        processed_text = selected_chunk

        response_schemas = [
            ResponseSchema(name="question", description="Вопрос с множественным выбором ответов, созданный на основе фрагмента входного текста."),
            ResponseSchema(name="option_1", description="Первый вариант ответа на вопрос с множественным выбором. Используйте этот формат: 'a) вариант ответа'"),
            ResponseSchema(name="option_2", description="Второй вариант ответа на вопрос с множественным выбором. Используйте этот формат: 'b) вариант ответа''"),
            ResponseSchema(name="option_3", description="Третий вариант ответа на вопрос с множественным выбором. Используйте этот формат: 'c) вариант ответа''"),
            ResponseSchema(name="answer", description="Правильный ответ на вопрос. Используйте этот формат: 'd) вариант ответа'' или 'b) вариант ответа'', и так далее.")
        ]

        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = output_parser.get_format_instructions()

        chat_model = ChatOpenAI(temperature=0, model_name='gpt-4o', api_key=st.secrets["OPENAI_API_KEY"])

        prompt = ChatPromptTemplate(
            messages=[
                HumanMessagePromptTemplate.from_template("""Получив текст нормативного документа, сгенерируй из него до пяти вопросов с несколькими вариантами ответов с правильным ответом.
                \n{format_instructions}\n{user_prompt}""")
            ],
            input_variables=["user_prompt"],
            partial_variables={"format_instructions": format_instructions}
        )

        user_query = prompt.format_prompt(user_prompt=processed_text)
        user_query_output = chat_model(user_query.to_messages())

        st.write("Сгенерированные вопросы:")
        st.write(user_query_output.content)