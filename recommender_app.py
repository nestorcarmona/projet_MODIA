import pickle
from BinaryClassifier import BinaryClassifier
import gradio as gr
import nltk
from nltk.corpus import stopwords
import numpy as np
import torch
import torch.nn as nn
from gensim.models.word2vec import Word2Vec
from pathlib import Path
import argparse
import warnings

warnings.filterwarnings(
    action="ignore",
    category=UserWarning,
)

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
    # Create an argument parser
    parser = argparse.ArgumentParser(
        description="Launch the recommender app"
    )
    parser.add_argument(
        "--word2vec_path",
        type=str,
        default="./checkpoints/word2vec.pkl",
        help="Path to the w2v model",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./checkpoints/binary_classifier.pth",
        help="Path to the binary_classifier model",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7878,
        help="Port to launch the app on",
    )
    # Parse the arguments
    args = parser.parse_args()
    port = args.port
    word2vec_path = Path(args.word2vec_path)
    model_path = Path(args.model_path)

    # Validate the word2vec model
    try:
        model_w2v = pickle.load(word2vec_path.open("rb"))
    except FileNotFoundError as e:
        print(f"Error: {word2vec_path} not found")
        raise e
    except Exception as e:
        print(f"Error: {word2vec_path} is not a valid pickle file")
        raise e
    # Validate the binary classifier model
    try:
        binary_classifier_weights = torch.load(model_path)
    except FileNotFoundError as e:
        print(f"Error: {model_path} not found")
        raise e
    except Exception as e:
        print(f"Error: {model_path} is corrupted")
        raise e
    
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
    

    # Load the model
    model = BinaryClassifier(input_shape=100, dropout=0.10)
    model.load_state_dict(binary_classifier_weights)
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
    demo.launch(share=True)