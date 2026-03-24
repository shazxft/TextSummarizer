from flask import Flask, render_template, request
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

app = Flask(__name__)

def summarize_text(text):
    stop_words = set(stopwords.words("english"))
    sentences = sent_tokenize(text,language='english')
    words = word_tokenize(text.lower())

    filtered_words = [word for word in words if word.isalnum() and word not in stop_words]

    # Frequency table
    freq_table = {}
    for word in filtered_words:
        freq_table[word] = freq_table.get(word, 0) + 1

    # Sentence scoring
    sentence_scores = {}
    for sentence in sentences:
        for word in freq_table:
            if word in sentence.lower():
                sentence_scores[sentence] = sentence_scores.get(sentence, 0) + freq_table[word]

        sentence_scores[sentence] = sentence_scores.get(sentence, 0) / len(word_tokenize(sentence))

    # Average score
    avg_score = np.mean(list(sentence_scores.values()))

    # Generate summary
    summary = ""
    for sentence in sentences:
        if sentence in sentence_scores and sentence_scores[sentence] > 1.2 * avg_score:
            summary += " " + sentence

    return summary


@app.route("/", methods=["GET", "POST"])
def index():
    summary = ""
    text = ""

    if request.method == "POST":
        text = request.form["text"]
        summary = summarize_text(text)

    return render_template("index.html", summary=summary, text=text)


if __name__ == "__main__":
    app.run(debug=True)
