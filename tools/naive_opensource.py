from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader
import dotenv
from langchain.memory import ConversationBufferMemory
dotenv.load_dotenv()

# Load, chunk and index the contents of the blog.
loader = TextLoader("../aipg.txt")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50, length_function=len)
splits = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
vectorDB = Chroma.from_documents(splits, embeddings)
memory = ConversationBufferMemory(memory_key="history", input_key="query", output_key='answer', return_messages=True)

input_variables = ['context', 'question']
template_str = "You are an assistant for question-answering tasks. Use the following pieces of retrieved information to answer the question consicely.\nQuestion: {question} \nContext: {context} \nAnswer:"
prompt_template = PromptTemplate(input_variables=input_variables, template=template_str)

llm = OpenAI()
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorDB.as_retriever(),
    chain_type_kwargs={"prompt": prompt_template},
    return_source_documents = True
)


#print(qa)
query = "Who is jjojo"
response = qa.invoke(query)

print(response)
print(f"Answer:\n===============================\n {response['result']}\n")
for doc in response['source_documents']:
    print(f"Source Document Content: {doc.page_content}\n")
