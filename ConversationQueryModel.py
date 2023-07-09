import os
import openai

from flask import Flask, request

from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.vectorstores import Chroma

app = Flask(__name__)
os.environ["OPENAI_API_KEY"] = "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"

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
    result = index.query(query)
    print(result)
    return result
    

if __name__ == '__main__':
    app.run()
