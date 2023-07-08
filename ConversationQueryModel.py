import os
import openai

from flask import Flask, request

from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma

app = Flask(__name__)
os.environ["OPENAI_API_KEY"] = "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"
loader = DirectoryLoader(".", glob="*.pdf")
index = VectorstoreIndexCreator().from_loaders([loader])
chat_history_dict = {}

# Enable to save to disk & reuse the model (for repeated queries on the same data)
PERSIST = True

#Train the Model
if PERSIST and os.path.exists("persist"):
  print("Reusing index...\n")
  vectorstore = Chroma(persist_directory="persist", embedding_function=OpenAIEmbeddings())
  index = VectorStoreIndexWrapper(vectorstore=vectorstore)
else:
  #loader = TextLoader("data/data.txt") # Use this line if you only need data.txt
  loader = DirectoryLoader("data/")
  if PERSIST:
    index = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory":"persist"}).from_loaders([loader])
  else:	
    index = VectorstoreIndexCreator().from_loaders([loader])

  chain = ConversationalRetrievalChain.from_llm(
  llm=OpenAI(),
  retriever=index.vectorstore.as_retriever(search_kwargs={"k": 3}),
)

# API call
@app.route('/process', methods=['POST'])
def process_input():
    data = request.get_json()
    print("data obtained is: ")
    print(data)
    text = data.get('text')
    userId = data.get('userId')

    print("text obtained is: ")
    print(text)
    # Pass the input to the model
    output = langchain_model(text, userId)

    # Return the model's response
    #print("Response: ")
    #print(output)
    # return jsonify({'output': output})
    return output


# Model training and generation
def langchain_model(query, userId):
    if chat_history_dict and chat_history_dict.keys and userId in chat_history_dict:
       chat_history = chat_history_dict[userId]
    else:
       chat_history = []
    if query in ['quit', 'q', 'exit']:
        chat_history = []
        chat_history_dict[userId] = chat_history
        return ""
    result = chain({"question": query, "chat_history": chat_history})
    print(result['answer'])
    chat_history.append(("Q:" +query, "A" +result['answer']))
    chat_history_dict[userId] = chat_history
    return result['answer']
    

if __name__ == '__main__':
    app.run()
