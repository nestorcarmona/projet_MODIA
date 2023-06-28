import pickle
from BinaryClassifier import BinaryClassifier
import gradio as gr
import nltk
from nltk.corpus import stopwords
import numpy as np
import torch
import torch.nn as nn
from gensim.models.word2vec import Word2Vec

def predict_from_text_wtv_batch(text: str) -> np.ndarray:
    text_tokenize = [nltk.word_tokenize(review) for review in text]
    text_tokenize = [[word.lower() for word in review if word.isalpha() and word not in stop_words] for review in text_tokenize]
    X = torch.from_numpy(np.stack([np.mean([model_w2v.wv[token] for token in review if token in model_w2v.wv] or [np.zeros(100)], axis=0) for review in text_tokenize])).to(device).float()
    model.eval()
    model.to(device)
    y_pred = model(X).cpu().detach().numpy()
    return np.stack([1-y_pred, y_pred])[..., 0].T

def predict_from_text_wtv(
        text: str, 
        model: nn.Module,
        model_w2v: Word2Vec,
        stop_words: list[str],
        ) -> np.ndarray:
    text = [text]
    text_tokenize = [nltk.word_tokenize(review) for review in text]
    text_tokenize = [
        [
            word.lower() for word in review 
            if word.isalpha() and word not in stop_words
        ]
        for review in text_tokenize
    ]
    X = torch.from_numpy(
        np.stack(
            [
                np.mean([model_w2v.wv[token] for token in review 
                if token in model_w2v.wv] or [np.zeros(100)], axis=0) 
                for review in text_tokenize
            ]
        )
    ).to(device).float()
    model.eval()
    model.to(device)
    y_pred = model(X).cpu().detach().numpy()[0][0]
    labels = {
        "Positive": float(y_pred),
        "Negative": float(1 - y_pred)
    }

    # # Explain the prediction
    # explainer = LimeTextExplainer(
    #     class_names=['Negative', 'Positive']
    # )

    # exp = explainer.explain_instance(
    #     text[0],
    #     predict_from_text_wtv_batch,
    #     num_features=20,
    # )

    # html = exp.as_html()
    # # html = (
    # #     ""
    # #     + html
    # #     + ""
    # # )
    # print(html)
    return labels

if __name__ == "__main__":
    # Define the device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Load the w2v model
    nltk.download('stopwords', quiet=True)
    stop_words = stopwords.words('english')
    model_w2v = pickle.load(open('./notebooks/model_w2v.pkl', 'rb'))

    # Load the model
    model = BinaryClassifier(input_shape=100, dropout=0.10)
    model.load_state_dict(torch.load('./notebooks/model_part2.pt'))
    model.eval()

    demo = gr.Interface(
        fn=lambda text: predict_from_text_wtv(
            text,
            model=model,
            model_w2v=model_w2v,
            stop_words=stop_words,
        ),
        inputs=gr.inputs.Textbox(
            lines=1,
            label="Text"
        ),
        outputs=[
            gr.outputs.Label(
                num_top_classes=2,
                label="Sentiment"
            ),
            # gr.outputs.HTML(
            #     label="Explanation"
            # )
        ],
        title="Sentiment Analysis",
        description="Predict the sentiment of a text",
        examples=[
            ["I love this recipe!"],
            ["I hate this recipe!"],
        ],
        allow_flagging="never",
    )
    demo.launch(share=False)