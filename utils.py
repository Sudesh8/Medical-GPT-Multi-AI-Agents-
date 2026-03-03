from dotenv import load_dotenv

load_dotenv("/content/my.env")
from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from typing_extensions import TypedDict
from langgraph.graph import START, END, StateGraph
from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from typing_extensions import TypedDict
from langgraph.graph import START, END, StateGraph
from langchain.pydantic_v1 import BaseModel
from langchain_groq.chat_models import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFaceEndpoint
from langchain_community.document_loaders import RecursiveUrlLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from tqdm import tqdm
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain_community.llms import Cohere

from dotenv import load_dotenv

load_dotenv("my.env")

import warnings

warnings.filterwarnings("ignore", category=UserWarning)


# Get API Keys securely
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
HF_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# check if keys loaded
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env file")

if not COHERE_API_KEY:
    raise ValueError("COHERE_API_KEY not found in .env file")

if not HF_API_TOKEN:
    raise ValueError("HUGGINGFACEHUB_API_TOKEN not found in .env file")

# Set environment variables
os.environ["COHERE_API_KEY"] = COHERE_API_KEY
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_API_TOKEN
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

HF_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
_llm = HuggingFaceEndpoint(repo_id=model_name)
# Loading Documents
# pdf_folder = "temp_dataset"
# splitter = RecursiveCharacterTextSplitter(
#     chunk_size=550, chunk_overlap=150, length_function=len
# )
# all_docs = []  # Store all document chunks
# for file in tqdm(os.listdir(pdf_folder)):
#     if file.endswith(".pdf"):  # Process only PDF files
#         file_path = os.path.join(pdf_folder, file)
#         loader = PyPDFLoader(file_path)

#         # Load raw documents
#         raw_docs = loader.load()

#         DOCS = splitter.split_documents(raw_docs)
#         all_docs.extend(DOCS)

# embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# # loading documents to vector db
# DB = FAISS.from_documents(all_docs, embed_model)


embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# load vector db
DATABASE_PATH = "DB"


DB = FAISS.load_local(DATABASE_PATH, embed_model, allow_dangerous_deserialization=True)


r_rerank = Cohere(temperature=0)
compressor = CohereRerank(model="rerank-english-v3.0")
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=DB.as_retriever()
)


## prompt

from langchain.prompts import PromptTemplate

# Query Parser
temp = """
yor are helpful bot . your task is to find key terms and topics for the given question.
i am working on medical documents so please create key terms and topics to this term only.
give in short .


question: {question}
key_terms_or_topics:
"""


# Document Reranker


# Answer
temp_rag = """
For given question find the answer given context and key_terms_or_topics.
if answer is not in the context just say i dont know
Synthesizes a concise answer from documents.
context: {context}
key_terms_or_topics:{key_terms_or_topics}
question: {question}

answer

"""


prompt_Qparser = PromptTemplate.from_template(temp)


prompt_rag = PromptTemplate.from_template(temp_rag)


# Loading LLMs

llm_Q = prompt_Qparser | _llm | StrOutputParser()
llm_rag = prompt_rag | _llm | StrOutputParser()


class AgentState(TypedDict):
    key_terms_or_topics: str
    init_msg: str
    context: str
    re_rank_context: str
    answer: str


workflow = StateGraph(AgentState)


print("\n\n ===============================")


def agent_Qkey_terms_or_topics(state):
    print("--- Start Qkey_terms_or_topics ---")
    question = state["init_msg"]
    key_terms_or_topics = llm_Q.invoke({"question": question})
    state["key_terms_or_topics"] = key_terms_or_topics
    return state


def agent_Doc_retriver(state):
    print("--- Agent Doc retriver ---")
    question = state["init_msg"]
    key_terms_or_topics = state["key_terms_or_topics"]
    final_qry = question + " " + key_terms_or_topics
    print("final_qry created")
    cont = DB.similarity_search(final_qry)
    state["context"] = cont
    return state


def agent_Document_Ranker(state):
    print("--- re_ranking started --- ")
    question = state["init_msg"]

    rerank_docs = compression_retriever.invoke(question)
    state["re_rank_context"] = rerank_docs
    return state


def agent_Response_Generator(state):
    print("---  Getting Answer ---")

    key_terms_or_topics = state["key_terms_or_topics"]
    init_msg = state["init_msg"]
    context = state["context"]
    re_rank_context = state["re_rank_context"]

    myinput = {
        "question": init_msg,
        "key_terms_or_topics": key_terms_or_topics,
        "context": re_rank_context,
    }

    answer = llm_rag.invoke(myinput)
    state["answer"] = answer

    return state


# def agent_prityprint(state):
#     print()
#     print(f"question {state['init_msg']}")
#     print("="*40)
#     print(f"answer {state['answer']}")


# node
workflow.add_node("agent_Qkey_terms_or_topics", agent_Qkey_terms_or_topics)


workflow.add_node("agent_Doc_retriver", agent_Doc_retriver)


workflow.add_node("agent_Document_Ranker", agent_Document_Ranker)

workflow.add_node("agent_Response_Generator", agent_Response_Generator)


# workflow.add_node("agent_prityprint",agent_prityprint)


# edge
workflow.add_edge(START, "agent_Qkey_terms_or_topics")

workflow.add_edge("agent_Qkey_terms_or_topics", "agent_Doc_retriver")

workflow.add_edge("agent_Doc_retriver", "agent_Document_Ranker")

workflow.add_edge("agent_Document_Ranker", "agent_Response_Generator")


workflow.add_edge("agent_Response_Generator", END)


graph = workflow.compile()


def chat(question):
    result = graph.invoke({"init_msg": question})
    return result["answer"]
