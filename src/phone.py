import os
from twilio.rest import Client

def verify_not_empty(parameter, error_message):
    if not parameter:
        raise ValueError(error_message)

def format_number(number,to_="E.164"):
    # format the number to E.164
    return number

def send_message(numbers, message):

    account_sid = os.environ['TWILIO_ACCOUNT_SID']
    auth_token = os.environ['TWILIO_AUTH_TOKEN']
    from_number = os.environ['TWILIO_PHONE_NUMBER']

    verify_not_empty(account_sid, "Account SID is not provided")
    verify_not_empty(auth_token, "Auth token is not provided")
    verify_not_empty(from_number, "TWILIO_PHONE_NUMBER is not provided")

    client = Client(account_sid, auth_token)

    for number in numbers:

        number = format_number(number,to_="E.164")
        message = client.messages.create(
            body=message,
            from_=from_number,
            to=number
        )

if __name__ == '__main__':
    send_message(
        ["111-222-3333", "222-333-4444", "333-444-5555", "123-456-7890", "987-654-3210",],
        "This is a Test message"
    )
