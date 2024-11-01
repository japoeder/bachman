import asyncio
import websockets
import json

async def test_client():
       uri = "ws://192.168.1.10:8002"
       async with websockets.connect(uri) as websocket:
           # Send parameters to the server
           params = {
               "ticker": "AAPL",
               "start_date": "2024-10-01T00:00:00-04:00",
               "end_date": "2024-10-06T23:59:00-04:00"
           }
           await websocket.send(json.dumps(params))

           # Receive and print data from the server
           try:
               while True:
                   data = await websocket.recv()
                   print("Received data:", data)
           except websockets.exceptions.ConnectionClosed:
               print("Connection closed")

# Run the test client
asyncio.run(test_client())