from flask import Flask, render_template, request, session
from services.rag_service import get_rag_response
import os

app = Flask(__name__)

app.secret_key = os.environ.get("FLASK_SECRET_KEY")


@app.route('/')
def home():

    session['messages'] = []

    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():

    prompt = request.form['prompt']

    # Get existing conversation
    messages = session.get("messages", [])

    response, sources = get_rag_response(prompt,messages)

    # Add current question and answer
    messages.append({'role': 'user','content': prompt})

    messages.append({'role': 'assistant','content': response,'sources': sources})

    session['messages'] = messages


    return render_template('result.html',messages=messages)


if __name__ == "__main__":
    app.run(debug=True, port=5001)