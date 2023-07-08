import os
import openai
import whisper
import io
from pydub import AudioSegment

from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma

app = Flask(__name__)
CORS(app, origins='*')
os.environ["OPENAI_API_KEY"] = "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"
chat_history_dict = {}

# Text
loader = DirectoryLoader("data/", glob="*.pdf")
index = VectorstoreIndexCreator().from_loaders([loader])

# Enable to save to disk & reuse the model (for repeated queries on the same data)
PERSIST = True

# Audio
model = whisper.load_model("base")

# Train the Model
if PERSIST and os.path.exists("persist"):
    print("Reusing index...\n")
    vectorstore = Chroma(persist_directory="persist", embedding_function=OpenAIEmbeddings())
    index = VectorStoreIndexWrapper(vectorstore=vectorstore)
else:
    loader = DirectoryLoader("data/")
    if PERSIST:
        index = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory": "persist"}).from_loaders([loader])
    else:
        index = VectorstoreIndexCreator().from_loaders([loader])

    chain = ConversationalRetrievalChain.from_llm(
        llm=OpenAI(),
        retriever=index.vectorstore.as_retriever(search_kwargs={"k": 3}),
    )


# Text API call
@app.route('/process-text', methods=['POST'])
def process_input():
    # Allow requests from any source
    response = jsonify({'result': 'success'})
    response.headers.add('Access-Control-Allow-Origin', '*')

    # Retrieve the text from the request
    data = request.get_json()
    text = data.get('text')
    user_id = data.get('user_id')

    # Pass the input to the model
    output = langchain_model(text, user_id)

    # Return the model's response
    return jsonify({'output': output})


# Model training and generation
def langchain_model(query, user_id):
    user_id = "SomeUserIdHere"
    if chat_history_dict and chat_history_dict.keys and user_id in chat_history_dict:
        chat_history = chat_history_dict[user_id]
    else:
        chat_history = []
    if query in ['quit', 'q', 'exit']:
        chat_history = []
        chat_history_dict[user_id] = chat_history
        return ""
    result = chain({"question": query, "chat_history": chat_history})
    print(result['answer'])
    chat_history.append(("Q:" + query, "A" + result['answer']))
    chat_history_dict[user_id] = chat_history
    return result['answer']


# Audio API calls
@app.route('/process-audio', methods=['POST'])
def process_audio():
    # Allow requests from any source
    response = jsonify({'result': 'success'})
    response.headers.add('Access-Control-Allow-Origin', '*')

    form_data = request.files['audio']
    print("formData received", form_data)
    audio_file = convert_audio(form_data)
    # user_id = request.form['userId']
    print("audio_file received", audio_file)

    file_path = "./output.mp3"

    # Output
    output = audio_model(file_path)

    user_id = "SomeUserId"

    # Sending text to langChain model
    output = langchain_model(output, user_id)
    # Delete file
    delete_file(file_path)

    return jsonify({'output': output})


def convert_audio(formData):
    # Read the binary data from the file
    audio_data = formData.read()

    # Create an in-memory file-like object
    audio_stream = io.BytesIO(audio_data)

    # Load the audio stream with pydub
    audio = AudioSegment.from_file(audio_stream)

    # Convert the audio to MP3
    audio = audio.set_frame_rate(44100)  # Set the desired frame rate
    audio = audio.set_channels(2)  # Set the desired number of channels
    audio.export('./output.mp3', format='mp3')

    return audio


# Model to convert audio to text
def audio_model(input_audio):
    openai.api_key = "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"

    print("Reached audio model")
    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(input_audio)
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # detect the spoken language
    _, probs = model.detect_language(mel)
    print(f"Detected language: {max(probs, key=probs.get)}")

    audio_file = open(input_audio, "rb")
    transcript = openai.Audio.translate("whisper-1", audio_file)

    return transcript.text


def delete_file(file_path):
    if file_path:
        try:
            # Delete the file
            os.remove(file_path)
            return jsonify({'message': 'File deleted successfully.'})
        except OSError as e:
            return jsonify({'message': 'Error occurred while deleting the file.'})
    else:
        return jsonify({'message': 'No file path provided.'})


if __name__ == '__main__':
    app.run(port=5002)
