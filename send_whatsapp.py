import os
from twilio.rest import Client

# Get from environment variables
account_sid = os.getenv("TWILIO_ACCOUNT_SID")
auth_token = os.getenv("TWILIO_AUTH_TOKEN")
from_whatsapp = os.getenv("TWILIO_WHATSAPP_FROM")
to_whatsapp = os.getenv("TWILIO_WHATSAPP_TO")

client = Client(account_sid, auth_token)

def send_whatsapp_message(msg: str):
    message = client.messages.create(
        body=msg,
        from_=f'whatsapp:{from_whatsapp}',
        to=f'whatsapp:{to_whatsapp}'
    )
    print("WhatsApp sent:", message.sid)
