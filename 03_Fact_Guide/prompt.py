from langchain.vectorstores.chroma import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

from redundant_filter_retriever import RedundantFilterRetriever
import langchain
langchain.debug = True

from dotenv import load_dotenv

load_dotenv()

chat = ChatOpenAI()

embeddings = OpenAIEmbeddings()
db = Chroma(
    persist_directory="emb",
    embedding_function=embeddings
)

# retriever = db.as_retriever()

retriever = RedundantFilterRetriever(
    embeddings=embeddings,
    chroma=db
)

chain = RetrievalQA.from_chain_type(
    llm=chat,
    retriever=retriever,
    chain_type="stuff"
)

# chain_type = "map_reduce"
# chain_type = "map_rerank"
# chain_type = "refine"

result = chain.run("What is an interesting fact about English language")
print(result)

