import telebot
import cv2 as cv
from PIL import Image
import requests
import io





teste_cod = False
CHAVE_API = "6181924385:AAEVTIvfL-aloqQC16l8PPcnpWfw2TSLScc"
bot = telebot.TeleBot(CHAVE_API)



def verificar_cod(mensagem):
    global teste_cod
    if teste_cod:
        if mensagem.text.isdigit():
            teste_cod = False
            return True
        else:
            bot.send_message(mensagem.chat.id,"Código incorreto! Por favor, digite apenas números.")




@bot.message_handler(commands=["sim"])
def sim(mensagem):
    global teste_cod
    bot.reply_to(mensagem,"Informe o código do dispositivo.")
    teste_cod = True

@bot.message_handler(commands=["cancelar"])    
def cancelar(mensagem):
    global teste_cod
    teste_cod = False
    bot.send_message(mensagem.chat.id,'Operação de localização cancelada. Digite oi, olá, ei, hey ou start para reiniciar a conversa.')


@bot.message_handler(func=verificar_cod)    
#@bot.message_handler(commands=["sim"])
def localizar(mensagem):
    # URL do servidor Flask
    url = 'http://192.168.1.7:5000/solicitar-foto/rascam_'+mensagem.text+'.png'

    # Enviar solicitação GET ao servidor

    response = requests.get(url)
    
    # Verificar a resposta do servidor
    print(response.status_code)
    if response.status_code == 500:
        global teste_cod
        bot.send_message(mensagem.chat.id,'Erro ao receber os dados do servidor. Verifique se o código está correto! Caso deseje, cancelar clique na opção abaixo: \n /cancelar')
        teste_cod = True
    elif response.status_code != 200:
        bot.send_message(mensagem.chat.id,'Erro ao enviar a solicitação de foto ao servidor.')
    else:


        image = response.content

        bot.send_message(mensagem.chat.id,"A ultima localização obtida será enviada a seguir, juntamente com a imagem adquirida do local")
        bot.send_photo(mensagem.chat.id,image)
        bot.send_message(mensagem.chat.id, "Deseja atualizar? Selecione a opção desejada: \n /sim         /nao")

@bot.message_handler(commands=["nao"])
def nao(mensagem):
    bot.send_message(mensagem.chat.id,"Tudo bem! Estarei aqui quando precisar!")

def verificar_1(mensagem):
    if mensagem.text.lower() in ["olá","oi","ei","hey","start","/start","ola","oi!","ei!","hey!","olá!","ola!"]:
        return True

def verificar_2(mensagem):
    if not(mensagem.text.lower() in ["olá","oi","ei","hey","start","/start","ola","oi!","ei!","hey!","olá!","ola!"]):
        return True

#def verificar_1(mensagem):
#    if mensagem.text.lower() =="olá" or mensagem.text.lower() == "oi" or mensagem.text.lower() == "ei" or mensagem.text.lower() == "start" or mensagem.text.lower()=="hey" or mensagem.text == "/start":
#        return True
    
#def verificar_2(mensagem):
#    if mensagem.text.lower() !="olá" and mensagem.text.lower() != "oi" and mensagem.text.lower() != "ei" and mensagem.text.lower() != "start" and mensagem.text.lower()!="hey" and mensagem.text != "/start":
#        return True

@bot.message_handler(func=verificar_1)
def resposta_inicial(mensagem):
    bot.reply_to(mensagem,"Olá, eu sou o localizador de entes queridos! Deseja iniciar a localização? Selecione a opção desejada:\n /sim       /nao")

@bot.message_handler(func=verificar_2)
def resposta_inapropriada(mensagem):
    bot.reply_to(mensagem,"Desculpe, não consegui entender! Digite oi, olá, ei, hey ou start para iniciarmos a conversa.")

bot.polling()