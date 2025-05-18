import nltk
import re
import torch
import joblib
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import wordnet
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline as hf_pipeline
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

class SentimentAwareTranslator:
    def __init__(self, idiom_csv_path, translation_model_path):
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
        # Set the device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
 
        self.idiom_df = pd.read_csv(idiom_csv_path, encoding='latin-1')
        self.translation_model_path = translation_model_path

        # Load your pretrained translation model (replace with your own)
        self.translator_tokenizer = AutoTokenizer.from_pretrained(translation_model_path)
        self.translator_model = AutoModelForSeq2SeqLM.from_pretrained(translation_model_path)

        # Move models to the same device
        self.translator_model = self.translator_model.to(self.device)

        # Load sentiment model
        self.sentiment_pipe = hf_pipeline("sentiment-analysis", model="cardiffnlp/twitter-xlm-roberta-base-sentiment")

    def _get_wordnet_pos(self, tag):
        if tag.startswith("J"):
            return wordnet.ADJ
        elif tag.startswith("V"):
            return wordnet.VERB
        elif tag.startswith("N"):
            return wordnet.NOUN
        elif tag.startswith("R"):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    def _lemmatize_sentence(self, sentence):
        lemmatizer = WordNetLemmatizer()
        tokens = word_tokenize(sentence)
        pos_tags = pos_tag(tokens)
        lemmas = [lemmatizer.lemmatize(w.lower(), self._get_wordnet_pos(pos)) for w, pos in pos_tags]
        return tokens, lemmas

    def _replace_idioms(self, sentence, idiom_df):
        original_tokens, sentence_lemmas = self._lemmatize_sentence(sentence)
        lemma_str = ' '.join(sentence_lemmas)

        for _, row in idiom_df.iterrows():
            idiom = row['idiom'].lower()
            meaning = row['english_meaning']

            idiom_tokens, idiom_lemmas = self._lemmatize_sentence(idiom)
            idiom_lemma_str = ' '.join(idiom_lemmas)

            if idiom_lemma_str in lemma_str:
                for i in range(len(sentence_lemmas) - len(idiom_lemmas) + 1):
                    if sentence_lemmas[i:i+len(idiom_lemmas)] == idiom_lemmas:
                        to_replace = ' '.join(original_tokens[i:i+len(idiom_lemmas)])
                        return re.sub(re.escape(to_replace), meaning, sentence, flags=re.IGNORECASE)
        return sentence


    def _analyze_sentiment(self, sentence):
        result = self.sentiment_pipe(sentence)[0]
        label = result['label']
        if label.startswith("1") or "Negative" in label:
            return "😠 Negative"
        elif label.startswith("3") or "Neutral" in label:
            return "😐 Neutral"
        else:
            return "🙂 Positive"

    def _translate(self, sentence):
        # Use sentence instead of text
        inputs = self.translator_tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}



        forced_bos_token_id = self.translator_tokenizer.convert_tokens_to_ids(["tam_Taml"])[0]

        # Provide a clear instruction for the model

        prompt = f"Translate the following text to Tamil while keeping words inside [ENTITY]...[/ENTITY] unchanged: {sentence}" # Changed 'text' to 'sentence'

        # Move prompt tensors to the same device as the model
        # Use inputs instead of prompt_inputs here.
        # Use input_ids and attention_mask, to make sure the input is correct.
        with torch.no_grad():
            outputs = self.translator_model.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], forced_bos_token_id=forced_bos_token_id)


        translated_text = self.translator_tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Define restore_named_entities inside translate_with_trained_model
        def restore_named_entities(translated_text):
            return translated_text.replace("[ENTITY]", "").replace("[/ENTITY]", "")

        return restore_named_entities(translated_text)

    def translate_with_sentiment(self, sentence):
        replaced = self._replace_idioms(sentence,self.idiom_df)
        sentiment = self._analyze_sentiment(replaced)
        translated = self._translate(replaced)
        return sentiment, translated