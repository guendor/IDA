import random
import numpy as np
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))


def clear_writing(writing):
    """
        Limpa todas as palavras inseridas.
    """
    # Tokeniza todas as frases inseridas, lematiza cada uma delas e retorna
    sentence_words = nltk.word_tokenize(writing)
    return [lemmatizer.lemmatize(word.lower()) for word in sentence_words]


# Retorna 0 ou 1 para cada palavra da bag de palavras.


def bag_of_words(writing, words):
    """
        Usa as sentenças que foram limpas para criar um pacote de
        palavras usadas para classes de previsão, baseadas nos
        resultados do treinamento do modelo.
    """
    # Tokenizando o pattern:
    sentence_words = clear_writing(writing)

    # Cria matriz de N palavras:
    bag = [0] * len(words)
    for sentence in sentence_words:
        for i, word in enumerate(words):
            if word == sentence:
                bag[i] = 1  # Atribui 1 no pacote de palavra se a palavra atual estiver na posição da frase.

    return np.array(bag)


def class_prediction(writing, model):
    """
        Prevê o pacote de palavras, usando como limite de erro 0.25,
        para evitar overfitting, e classifica os resultados por
        força da probabilidade.
    """
    # Filtrar previsões abaixo de um limite de 0.25
    prevision = bag_of_words(writing, words)
    response_prediction = model.predict(np.array([prevision]))[0]
    results = [[index, response] for index, response in
               enumerate(response_prediction) if response > 0.25]

    # Verifica nas previsões se há 1 na lista, se não há envia resposta
    # padrão (anything_else) ou se não corresponde a margem de erro.
    if "1" not in str(prevision) or len(results) == 0:
        results = [[0, response_prediction[0]]]

    # Classifica por força probabilística:
    results.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": classes[r[0]], "probability":
             str(r[1])} for r in results]


def get_response(intents, intents_json):
    """
        A partir da lista gerada, verifica o arquivo json e produz a
        maioria das respostas com a maior probabilidade.
    """
    tag = intents[0]['intent']
    list_of_intents = intents_json['intents']
    for idx in list_of_intents:
        if idx['tag'] == tag:
            # Caso as respostas sejam um array com várias opções,
            # pegamos uma resposta randômica da lista.
            result = random.choice(idx['responses'])
            break
    return result
