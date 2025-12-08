from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pprint
path = "./Data/1312.4400v3.pdf"
loader = PyPDFLoader(path)

doc = loader.load()

print(len(doc)) # pages count

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    length_function=len,
)

documents = text_splitter.split_documents(doc)
pprint.pp(documents[0].metadata)
pprint.pp(documents[0].page_content)

