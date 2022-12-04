# Python class: QaEngine
# Description: This class is used to answer questions based on pre-trained models and pre-processed documents.
# References:
# 1. Deepset Blog: https://haystack.deepset.ai/tutorials/01_basic_qa_pipeline
# 2. Deepset Github: https://github.com/deepset-ai/haystack

import os
import re

from haystack.document_stores import ElasticsearchDocumentStore
from haystack.nodes import BM25Retriever, FARMReader
from haystack.pipelines import ExtractiveQAPipeline
from haystack.utils import clean_wiki_text, convert_files_to_docs, print_answers, launch_es


class Prediction:
    def __init__(self, answer, score, context, extracted_answer):
        self.answer = answer
        self.score = score
        self.context = context
        self.extracted_answer = extracted_answer


class QAEngine:
    def __init__(self, model_name_or_path='deepset/roberta-base-squad2', host='localhost', port=9200, index='document'):
        self.model_name_or_path = model_name_or_path
        self.host = host
        self.port = port
        self.index = index

        # Step 1: Launch an Elasticsearch instance locally
        launch_es()

        self.host = os.environ.get("ELASTICSEARCH_HOST", "localhost")
        self.document_store = ElasticsearchDocumentStore(host=self.host, username="", password="", index=self.index)
        self.retriever = BM25Retriever(document_store=self.document_store)
        self.reader = FARMReader(model_name_or_path=self.model_name_or_path, use_gpu=True)
        self.pipe = ExtractiveQAPipeline(reader=self.reader, retriever=self.retriever)

    def extract_sentence(self, text, word):
        pattern = r"\b[A-Z][^\.!?]*\b" + word + r"\b[^\.!?]*[\.!?]"
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return matches[0]
        return ''

    def index_documents(self):
        # Step 1: Output file to be written to a directory.
        doc_dir = "data/uploaded_docs"

        # Step 2: Convert files to dicts
        # You can optionally supply a cleaning function that is applied to each doc (e.g. to remove footers)
        # It must take a str as input, and return a str.
        docs = convert_files_to_docs(dir_path=doc_dir, clean_func=clean_wiki_text, split_paragraphs=True)

        # We now have a list of dictionaries that we can write to our document store. If your texts come from a
        # different source (e.g. a DB), you can of course skip convert_files_to_dicts() and create the dictionaries
        # yourself. The default format here is: { 'content': "<DOCUMENT_TEXT_HERE>", 'meta': {'name':
        # "<DOCUMENT_NAME_HERE>", ...} } (Optionally: you can also add more key-value-pairs here, that will be
        # indexed as fields in Elasticsearch and can be accessed later for filtering or shown in the responses of the
        # Pipeline)

        # Step 3: Now, let's write the dicts containing documents to our DB.
        self.document_store.write_documents(docs)

    def answer_question(self, query):
        predictions = self.reader.predict(query=query, documents=self.document_store.get_all_documents())

        ans_predictions = []

        # Iterate over the answers and print them
        for answer in predictions['answers']:
            extracted_sentence = self.extract_sentence(answer.context, answer.answer)
            if extracted_sentence:
                ans_predictions.append(Prediction(answer.answer, answer.score, answer.context, extracted_sentence))
            else:
                ans_predictions.append(Prediction(answer.answer, answer.score, answer.context, answer.answer))

        return ans_predictions
