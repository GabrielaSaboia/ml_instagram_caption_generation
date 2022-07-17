import os
from twilio.rest import Client
from flask import Flask, request

account_sid = os.environ['TWILIO_ACCOUNT_SID']
auth_token = os.environ['TWILIO_AUTH_TOKEN']
client = Client(account_sid, auth_token)

app = Flask(__name__)


@app.route("/get_image", methods=['POST'])
def receive_image():
    NumMedia = request.values.get('NumMedia')
    from_wa = request.values.get('From')
    to_wa = request.values.get('To')
    if NumMedia != "0":
        img_url = request.values.get('MediaUrl0')
        caption = generate_caption(img_url)
        message = client.messages.create(
            body=f"Suggested Caption: \n {caption}",
            from_=f'whatsapp:{to_wa}',
            to=f'whatsapp:{from_wa}'
        )
    else:
        message = client.messages.create(
            body='Please upload an image to get caption suggestions.',
            from_=f'whatsapp:{to_wa}',
            to=f'whatsapp:{from_wa}'
        )
    return "Success", 200


def generate_caption(img_url) -> str:
    pass


if __name__ == '__main__':
    app.run(debug=True, port=8000)
