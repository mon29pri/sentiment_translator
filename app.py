from flask import Flask, render_template, request
from SentimentAwareTranslator import SentimentAwareTranslator

app = Flask(__name__)

# Load model once (avoid reloading for every request)
pipeline = SentimentAwareTranslator(
    idiom_csv_path="Idiom_dataset.csv",
    translation_model_path="Mona29pri/Translator"
)

def get_emoji(sentiment):
    return {
        'positive': '😊',
        'negative': '😠',
        'neutral': '😐'
    }.get(sentiment, '🙂')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
def translate():
    input_text = request.form['input_text']
    sentiment, translation = pipeline.translate_with_sentiment(input_text)
    emoji = get_emoji(sentiment)
    
    return render_template(
        'result.html',
        input_text=input_text,
        translated_text=translation,
        sentiment=sentiment,
        emoji=emoji
    )

if __name__ == '__main__':
    app.run(debug=True)
