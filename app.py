import os
from pathlib import Path
import joblib
import pandas as pd
import gradio as gr

model = joblib.load(Path(__file__).resolve().parent / 'creditability_model.joblib')

def predict(age, credit_amount):
    inputs = pd.DataFrame([[age, credit_amount]], columns=['Age', 'Credit Amount'])
    predicted = int(model.predict(inputs)[0])
    probability = float(model.predict_proba(inputs)[0][list(model.classes_).index(1)])
    label = 'Good Creditability' if predicted == 1 else 'Bad Creditability'
    return label, f'{probability:.1%}'

with gr.Blocks(title='Creditability Prediction') as demo:
    gr.Markdown('# Creditability Prediction')
    gr.Markdown('Educational demonstration only — not for actual credit decisions.')
    age = gr.Number(label='Age', value=35, minimum=18, maximum=100, precision=0)
    amount = gr.Number(label='Credit Amount', value=3000, minimum=0)
    button = gr.Button('Predict Creditability', variant='primary')
    result = gr.Textbox(label='Prediction')
    confidence = gr.Textbox(label='Estimated Probability of Good Creditability')
    button.click(predict, inputs=[age, amount], outputs=[result, confidence])

if __name__ == '__main__':
    demo.launch(server_name='0.0.0.0', server_port=int(os.environ.get('PORT', 7860)))
