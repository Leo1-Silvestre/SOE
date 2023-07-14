from flask import Flask, request, send_file
from PIL import Image
import io
import requests
import base64

def get_encoded_img(image_path):
    img = Image.open(image_path, mode='r')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG') 
    return base64.encodebytes(img_byte_arr.getvalue()).decode('ascii')



app = Flask(__name__)

@app.route('/enviar-foto', methods=['POST'])
def receber_foto():
    
    # Recebe os dados da imagem e do código enviados pelo cliente
    data = request.data

    # Separa o código dos dados da imagem
    codigo, image_data = data.split(b'\0', 1)

   
    file_name = "rascam_"+codigo+".png"
    # Salva a imagem em um arquivo
    with open(file_name, 'wb') as file:
        file.write(image_data)

    # Retorna uma resposta de sucesso para o cliente
    return 'Imagem recebida e salva com sucesso!'

@app.route('/solicitar-foto/<path:path>', methods=['POST', 'GET'])
def enviar_foto(path):
    #foto = request.files['foto']
    #foto.save('/caminho/para/salvar/a/foto.jpg')

    #Identifica quem solicitou a imagem
    chat_id = request.args.get('chat_id')
    print(chat_id, " Esta falando comigo")

        

    return send_file(path, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
