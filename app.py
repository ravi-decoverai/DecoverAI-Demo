from io import StringIO

import streamlit as st
import os
import pandas as pd
import sqldf

from qa import get_query_from_openai, Table
from qa_engine import QAEngine

# Upload the file using the file_uploader function.
# It should accept csv, txt and pdf files.
uploaded_file = st.file_uploader("Choose a file", type=["csv", "txt", "pdf"])

# Show a dropdown to select a set of CSV files.
# The files are stored in the data folder.
# The user should also be able to upload a new file.
# The user should be able to select a table from the uploaded file.
# The first open should be empty file.

csv_files = os.listdir("data")
# Remove any directories from the list
csv_files = [file for file in csv_files if os.path.isfile(os.path.join("data", file))]
# Add an empty option to the list
csv_files.insert(0, "Select from the list here")
selected_csv_file = st.selectbox("Select a CSV file", csv_files)

if selected_csv_file == "Upload your own file":
    selected_csv_file = uploaded_file

df = None

if selected_csv_file != "Select from the list here":
    df = pd.read_csv("data/" + selected_csv_file)

if uploaded_file is not None:
    # Detect if the file is a CSV file
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)

    # Detect if the file is a txt file
    if uploaded_file.name.endswith(".txt"):
        # To convert to a string based IO:
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))

        # To read file as string:
        string_data = stringio.read()
        with open("data/uploaded_docs/" + uploaded_file.name, "w") as f:
            f.write(string_data)

    if uploaded_file.name.endswith(".pdf"):
        bytes_data = uploaded_file.getvalue()

        with open("data/uploaded_docs/" + uploaded_file.name, "wb") as f:
            f.write(bytes_data)
    # Instantiating the QAEngine
    qa = QAEngine()
    qa.index_documents()

    # Make a call to OpenAI's API to get the answer to the question.
    # The answer is a string.
    # Detect if the user has pressed enter on th text box.
    # If the user has pressed enter, then call the OpenAI API.
    # If the user has not pressed enter, then do not call the OpenAI API.
    question = st.text_input("Ask a question to be answered from the uploaded document", "")
    if st.button("Get Answer"):
        predictions = qa.answer_question(question)
        # Check if the array is empty
        if predictions:
            extracted_answer = predictions[0].extracted_answer
            st.write(extracted_answer)

if df is not None:
    # Display only the first 5 rows of the DataFrame
    st.write(df.head())

    # Extract the first line as a list of column names
    column_names = df.columns.tolist()

    # For each column replace the spaces with underscores
    column_names = [column_name.replace(" ", "_") for column_name in column_names]
    df.columns = column_names

    # Show a text input to ask the user a question
    question = st.text_input("Ask a question about the data", "")

    # Translate the column names to a single table
    table = Table('df', column_names)

    tables = [table]

    # Make a call to OpenAI's API to get the answer to the question.
    # The answer is a string
    # Detect if the user has pressed enter on th text box.
    # If the user has pressed enter, then call the OpenAI API.
    # If the user has not pressed enter, then do not call the OpenAI API.
    if st.button("Get Answer From The Table"):
        query = get_query_from_openai(question, tables)

        # Replace table with the word df
        query = query.replace("table", "df")

        # Run the SQL query on the DataFrame and display the results
        st.write(sqldf.run(query))
