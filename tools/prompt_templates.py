from langchain.prompts import PromptTemplate

input_variables = ['context', 'question']
template_str = "You are an assistant for question-answering tasks. Use the following pieces of retrieved information to answer the question consicely.\nQuestion: {question} \nContext: {context} \nAnswer:"
rag_prompt_template = PromptTemplate(input_variables=input_variables, template=template_str)