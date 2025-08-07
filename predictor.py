from dataclasses import dataclass
from typing import List, Tuple
import re
from pathlib import Path
import pickle
from tensorflow import keras
import nltk
from nltk.tokenize import word_tokenize
import numpy as np

from config import Config

nltk.download('punkt_tab')

@dataclass
class NerPredictor:
    model_path: Path
    word2idx_path: Path
    idx2tag_path: Path
    max_len: int

    def __post_init__(self) -> None:
        self.model = keras.models.load_model(self.model_path)

        with open(self.word2idx_path, "rb") as w2i_f:
            self.word2idx = pickle.load(w2i_f)
        with open(self.idx2tag_path, "rb") as i2t_f:
            self.idx2tag = pickle.load(i2t_f)

    @staticmethod
    def tokenize_sentence(sentence: str) -> List[str]:
        text = re.sub(r"[`''ʻʼ']s\b", "", sentence)
        text = re.sub(r"[`''ʻʼ']", "", text)
        text = re.sub(r'[""„"«»]', "", text)
        tokens = word_tokenize(text.lower())
        return tokens

    def encode_data(self, sentence: str) -> Tuple[List[int], List[str]]:
        tokens = self.tokenize_sentence(sentence)
        word_indices = []

        # words to indexes
        for word in tokens:
            word_indices.append(
                self.word2idx.get(word, self.word2idx['<UNK>'])
            )

        # padding
        if len(word_indices) > self.max_len:
            word_indices = word_indices[:self.max_len]
            tokens = tokens[:self.max_len]
        else:
            padding_length = self.max_len - len(word_indices)
            word_indices.extend([0] * padding_length)

        return word_indices, tokens
    
    def get_predictions(self, texts_list: List[str]) -> Tuple[List[List[Tuple[str, str]]], List[List[str]]]:
        tokens_list = []
        word_indices_list = []

        for text in texts_list:
            word_indices, tokens = self.encode_data(sentence=text)
            tokens_list.append(tokens)
            word_indices_list.append(word_indices)
            
        input_data = np.array(word_indices_list)
        predictions = self.model.predict(input_data)

        all_predicted_indices = np.argmax(predictions, axis=-1)  # (batch_size, seq_len)

        all_results = []
        for ind, tokens in enumerate(tokens_list):
            # Pobierz tagi dla tej sekwencji
            predicted_indices = all_predicted_indices[ind]
            predicted_tags = [self.idx2tag[idx] for idx in predicted_indices]
            
            # Filtruj PAD tokens
            results = [(token, tag) for token, tag in zip(tokens, predicted_tags) if tag != '<PAD>']
            all_results.append(results)

        all_results = []
        for ind, tokens in enumerate(tokens_list):
            predicted_tags = []
            for i, token in enumerate(tokens):
                # if i < len(tokens): # to sprawdzic
                tag_idx = np.argmax(predictions[ind][i])
                tag = self.idx2tag[tag_idx]
                predicted_tags.append(tag)

            results = []
            for token, tag in zip(tokens, predicted_tags):
                if tag != '<PAD>':
                    results.append((token, tag))
            all_results.append(results)

        return all_results, tokens_list


if __name__ == "__main__":
    ner_pred = NerPredictor(
        model_path=r"C:\Users\table\PycharmProjects\MojeCos\ner_fajny\models\model1\ner_model_2025_07_13.h5",
        word2idx_path=r"C:\Users\table\PycharmProjects\MojeCos\ner_fajny\models\model1\word2idx_2025_07_13.pkl",
        idx2tag_path=r"C:\Users\table\PycharmProjects\MojeCos\ner_fajny\models\model1\idx2tag_2025_07_13.pkl",
        max_len=Config.MAX_LEN
    )
    x, y = ner_pred.get_predictions(
        texts_list=["The FBI's has opened an investigation against former FBI and CIA directors. Russian interference in the US election is in the background. Conference will start at 20.07.2025"]
    )
    print(x)
    print()
    print(y)
    # test_cases = ['Adam"s', "Adam's", 'Adam`s', "Adam's", 'He said "hello"']
    # print(ner_pred.tokenize_sentence('Adam"s'))
    # for c in test_cases:
    #     print(f"{c} -> {ner_pred.tokenize_sentence(c)}")
    # x = ner_pred.get_prediction(
    #     text="The FBI has opened an investigation against former FBI and CIA directors. Russian interference in the US election is in the background. Conference will start at 20.07.2025"
    # )
    # print(x)