import streamlit as st
from pdfminer.high_level import extract_text
from langchain.schema import HumanMessage
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_openai import ChatOpenAI
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_core.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
import string
import re
import docx

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

st.title("Упроститель текста и генератор вопросов")
uploaded_file = st.file_uploader("Загрузите DOCX файл", type=["docx"])

if uploaded_file is not None:
    # Use a library like python-docx to extract text from DOCX files
    def extract_text_from_docx(file):
        doc = docx.Document(file)
        return '\n'.join([para.text for para in doc.paragraphs])

    full_text = extract_text_from_docx(uploaded_file)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=5000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_text(full_text)
    chunk_index = st.selectbox("Выберите чанк", range(len(chunks)))
    selected_chunk = chunks[chunk_index]
    st.write("Выбранный чанк:")
    st.write(selected_chunk)
    chat_model = ChatOpenAI(temperature=0, model_name='gpt-4o', api_key=st.secrets["OPENAI_API_KEY"])

    if 'simplify_chunk' not in st.session_state:
        st.session_state.simplify_chunk = False

    if 'generate_questions' not in st.session_state:
        st.session_state.generate_questions = False

    if st.button("Упростить чанк"):
        st.session_state.simplify_chunk = True
        st.session_state.generate_questions = False

    if st.button("Сгенерировать вопросы по чанку"):
        st.session_state.generate_questions = True
        st.session_state.simplify_chunk = False

    if st.session_state.simplify_chunk:
        doc = [Document(page_content=selected_chunk)]
        prompt_template = """Перепиши текст, упростив его, при этом не используй нумерацию абзацев:
        "{text}".
        """
        prompt = PromptTemplate.from_template(prompt_template)
        llm_chain = LLMChain(llm=chat_model, prompt=prompt)
        stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")
        selected_chunk_summary = stuff_chain.run(doc)
        st.write(selected_chunk_summary)

    if st.session_state.generate_questions:
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