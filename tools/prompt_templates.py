from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader
import dotenv

dotenv.load_dotenv()

# Load, chunk and index the contents of the blog.
loader = TextLoader("../aipg.txt")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50, length_function=len)
splits = text_splitter.split_documents(documents)


prompt_template = """You are a Chat customer support agent.
        Address the customer as Dear Mr. or Miss. depending on customer's gender followed by Customer's First Name.
        Use the following customer related information (delimited by <cp></cp>) context (delimited by <ctx></ctx>) and the chat history (delimited by <hs></hs>) to answer the question at the end:
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Below are the details of the customer:\n 
        <cp>
        Customer's Name: {Customer_Name}
        Customer's Resident State: {Customer_State}
        Customer's Gender: {Customer_Gender}
        </cp>
        <ctx>
        {context}
        </ctx>
        <hs>
        {history}
        </hs>
        Question: {query}
        Answer: """

# print(prompt_template.format(cProfile))

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["history", "context", "query", "Customer_Name", "Customer_State", "Customer_Gender"]
)

embeddings = OpenAIEmbeddings()
vectorDB = Chroma.from_documents(splits, embeddings)

from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(memory_key="history", input_key="query", output_key='answer', return_messages=True)

qa = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    chain_type='stuff',
    retriever=vectorDB.as_retriever(),
    verbose=True,
    chain_type_kwargs={
        "verbose": True,
        "prompt": PROMPT,
        "memory": memory
    }
)

print(qa)

# qa({"query": "who's the client's friend?", "Customer_Gender": "Male", "Customer_State": "New York",
#     "Customer_Name": "Aaron"})
#
#
# qa({"query": "who's jojo?", "Customer_Gender": "Male", "Customer_State": "New York",
#     "Customer_Name": "Aaron"})