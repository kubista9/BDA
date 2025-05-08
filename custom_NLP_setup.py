from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# 1. Load and split your text
loader = TextLoader("my_corpus.txt")
documents = loader.load()
splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = splitter.split_documents(documents)

# 2. Create vector database
embeddings = OpenAIEmbeddings()
db = Chroma.from_documents(docs, embeddings)

# 3. Ask questions
qa = RetrievalQA.from_chain_type(llm=OpenAI(), retriever=db.as_retriever())
print(qa.run("What are the main themes in the text?"))
