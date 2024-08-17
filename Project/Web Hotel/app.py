from flask import Flask, render_template, request
import torch
from transformers import BertTokenizer, BertForSequenceClassification

app = Flask(__name__)

model_name = "bert-base-uncased"
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=34*4)
tokenizer = BertTokenizer.from_pretrained(model_name)
model.eval()

categories = ['FACILITIES#CLEANLINESS', 'FACILITIES#COMFORT', 'FACILITIES#DESIGN&FEATURES', 
              'FACILITIES#GENERAL', 'FACILITIES#MISCELLANEOUS', 'FACILITIES#PRICES', 
              'FACILITIES#QUALITY', 'FOOD&DRINKS#MISCELLANEOUS', 'FOOD&DRINKS#PRICES', 
              'FOOD&DRINKS#QUALITY', 'FOOD&DRINKS#STYLE&OPTIONS', 'HOTEL#CLEANLINESS', 
              'HOTEL#COMFORT', 'HOTEL#DESIGN&FEATURES', 'HOTEL#GENERAL', 'HOTEL#MISCELLANEOUS', 
              'HOTEL#PRICES', 'HOTEL#QUALITY', 'LOCATION#GENERAL', 'ROOM_AMENITIES#CLEANLINESS', 
              'ROOM_AMENITIES#COMFORT', 'ROOM_AMENITIES#DESIGN&FEATURES', 'ROOM_AMENITIES#GENERAL', 
              'ROOM_AMENITIES#MISCELLANEOUS', 'ROOM_AMENITIES#PRICES', 'ROOM_AMENITIES#QUALITY', 
              'ROOMS#CLEANLINESS', 'ROOMS#COMFORT', 'ROOMS#DESIGN&FEATURES', 'ROOMS#GENERAL', 
              'ROOMS#MISCELLANEOUS', 'ROOMS#PRICES', 'ROOMS#QUALITY', 'SERVICE#GENERAL']

def predict(model, tokenizer, sentence):
    inputs = tokenizer.encode_plus(
        sentence,
        max_length=128,
        add_special_tokens=True,
        return_token_type_ids=True,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    token_type_ids = inputs['token_type_ids']

    with torch.no_grad():
        outputs = model(input_ids, attention_mask, token_type_ids)
        predictions = torch.sigmoid(outputs.logits).cpu().numpy()

    return predictions

def decode_predictions(predicted_labels, threshold=0.5):
    sentiments = ['positive', 'negative', 'neutral', 'none']
    decoded_predictions = {}
    for i, pred in enumerate(predicted_labels[0]):
        if pred > threshold:
            sentiment = sentiments[i % 4]
            if sentiment in ['neutral', 'none']:
                continue
            category_index = i // 4
            category = categories[category_index]
            if category not in decoded_predictions or pred > decoded_predictions[category][1]:
                decoded_predictions[category] = (sentiment, pred)
    
    return [f"{category}: {sentiment}" for category, (sentiment, _) in decoded_predictions.items()]

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict_route():
    predictions = []
    if request.method == 'POST':
        input_text = request.form['input_text']
        predicted_labels = predict(model, tokenizer, input_text)
        predictions = decode_predictions(predicted_labels)
    return render_template('predict.html', predictions=predictions)

@app.route('/airline')
def airline():
    return render_template('airline.html')

@app.route('/review')
def review():
    return render_template('review.html')

@app.route('/HCM')
def HCM():
    return render_template('HCM.html')

@app.route('/HL')
def HL():
    return render_template('HL.html')

@app.route('/DN')
def DN():
    return render_template('DN.html')

@app.route('/CT')
def CT():
    return render_template('CT.html')

@app.route('/NT')
def NT():
    return render_template('NT.html')


if __name__ == '__main__':
    app.run(debug=True)
