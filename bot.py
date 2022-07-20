import json
import tkinter
from tkinter import *
from extract import class_prediction, get_response
from keras.models import load_model

# Extrai o modelo através do keras. Em seguida carrega as intents:
model = load_model('model.h5')
intents = json.loads(open('intents.json').read())

base = Tk()
base.title("Interface Dialética de Aprendizado - IDA")
base.geometry("410x500")
base.resizable(width=FALSE, height=FALSE)


def chatbot_response(msg):
    """
        Resposta da IDA.
    """
    ints = class_prediction(msg, model)
    res = get_response(ints, intents)
    return res


def send():
    """
        Envia a mensagem.
    """
    msg = EntryBox.get("1.0", 'end-1c').strip()
    EntryBox.delete("0.0", END)

    if msg != '':
        Chat.config(state=NORMAL)
        Chat.insert(END, f"Você: {msg}\n\n")
        Chat.config(foreground="#000000", font=("Arial", 12))

        response = chatbot_response(msg)
        Chat.insert(END, f"Bot: {response}\n\n")

        Chat.config(state=DISABLED)
        Chat.yview(END)


# Janela do Chat
Chat = Text(base, bd=0, bg='white', height='8', width='50', font='Arial',)
Chat.config(state=DISABLED)

# Vincula a barra de rolagem à janela de bate papo.
scrollbar = Scrollbar(base, command=Chat.yview)
Chat['yscrollcommand'] = scrollbar.set

# Botão de envio de mensagem, comando envia para a função de send.
SendButton = Button(base, font=('Verdana', 10, 'bold'), text='Enviar',
                    width='12', height=2, bd=0, bg='#666', activebackground='#333',
                    fg='#ffffff', command=send)

# Cria a Box de texto:
EntryBox = Text(base, bd=0, bg='white', width='29', height='2', font='Arial')

# Joga todos os componentes na tela:
scrollbar.place(x=386, y=6, height=386)
Chat.place(x=6, y=6, height=386, width=380)
EntryBox.place(x=128, y=401, height=50, width=260)
SendButton.place(x=6, y=401, height=50)


base.mainloop()
