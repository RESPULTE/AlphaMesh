import websocket


def on_message(ws, message):
    print(message)


ws = websocket.WebSocketApp(
    "wss://ws.finnhub.io?token=d48rf01r01qnpsnp6dqgd48rf01r01qnpsnp6dr0",
    on_message=on_message,
)
ws.run_forever()
