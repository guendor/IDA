import json
import pickle
import nltk
import random
import numpy as np
import nltk
nltk.download('omw-1.4')

from nltk.stem import WordNetLemmatizer
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD

nltk.download('punkt')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

# Iniciar lista de palavras, classes, docs e palavras ignoradas:
words = []
documents = []
intents = json.loads(open('intents.json').read())

# Adicionar as tags na lista de classes:
classes = [i['tag'] for i in intents['intents']]
ignore_words = ["!", "@", "#", "$", "%", "*", "?"]

# Leitura do arquivo intents.json e transformação em json:
intents = json.loads(open('intents.json').read())

# Percorre o array de objetos:
for intent in intents['intents']:
    for pattern in intent['patterns']:
        word = nltk.word_tokenize(pattern)  # Tokenização dos patterns.
        words.extend(word)  # Adição do token na lista de palavras.
        documents.append((word, intent['tag']))  # Adição aos documentos para identificação da tag.

# Lematizar palavras ignorando as palavras da lista ignore_words.
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]

# Classificação das listas:
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

# Salvar as palavras e classes nos arquivos pkl:
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# --! Deep Learning !-- #

# Inicializando o treinamento:
training = []
output_empty = [0] * len(classes)
for document in documents:
    bag = []  # Inicialização do saco de palavras
    pattern_words = document[0]  # Listagem de palavras do pattern

    # Lematização de cada palavra na tentativa de representar palavras relacionadas:
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]

    # Cria conjunto de palavras com 1, se encontrada correspondência de palavras no padrão atual:
    for word in words:
        bag.append(1) if word in pattern_words else bag.append(0)

    # output_row como chave para a lista, onde saida será 0 para cada tag e 1 para a tag atual.
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1

    training.append([bag, output_row])

# Embaralhamento do conjunto de treinamentos e transformação em numpy array
random.shuffle(training)
training = np.array(training)

# Lista de treino sendo x == patterns e y == intenções
x = list(training[:, 0])
y = list(training[:, 1])

# Cria o modelo com 3 camadas
# Primeira camada de 128 neurônios,
# Segunda camada de 64 neurônios
# Terceira camada de saída
# N° de neurônios igual ao n° de intenções para prever intenção de saída com softmax

model = Sequential()
model.add(Dense(128, input_shape=(len(x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(y[0]), activation='softmax'))

# Modelo é compilado com DESCIDA DE GRADIENTE ESTOCÁSTICA -> ler docs!
# Com gradiente acelerado de Nesterov.
# Nesterov Accelerated Gradient (NAG) -> Mede o gradiente da função de custo, NÃO na posição local,
# mas um tanto à frente na direção do momentum.
# A única diferença entre a otimização de Momentum é que o gradiente é medido em θ + βm em vez de em θ.

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Ajuste e salve do modelo:
m = model.fit(np.array(x), np.array(y), epochs=200, batch_size=5, verbose=1)
model.save('model.h5', m)
