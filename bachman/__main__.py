import asyncio
import websockets
import json
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()
from bachman._utils.mongo_conn import mango_conn
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

async def stream_data(websocket, path):
    try:
        logging.info("Client connected")
        
        # Receive parameters from the client
        params = await websocket.recv()
        logging.debug(f"Received parameters: {params}")
        params = json.loads(params)
        ticker = params.get('ticker')
        
        # Update the format string to match the full date-time format
        start_date = datetime.strptime(params.get('start_date'), '%Y-%m-%dT%H:%M:%S%z')
        end_date = datetime.strptime(params.get('end_date'), '%Y-%m-%dT%H:%M:%S%z')

        logging.info(f"Querying data for ticker: {ticker}, from {start_date} to {end_date}")

        # Connect to MongoDB
        db = mango_conn()
        collection = db['historicalPrices']

        # Query the database
        query = {
            "ticker": ticker,
            "timestamp": {"$gte": start_date, "$lte": end_date}
        }
        cursor = collection.find(query).sort("date", 1)

        # Stream data to the client
        for document in cursor:
            logging.debug(f"Sending document: {document}")
            await websocket.send(json.dumps(document))
            await asyncio.sleep(1)  # Simulate data arriving every second
    except Exception as e:
        logging.error(f"Error: {e}")
    finally:
        logging.info("Closing connection")
        await websocket.close()

async def main():
    logging.info("Starting WebSocket server on port 8002")
    server = await websockets.serve(stream_data, "0.0.0.0", 8002)
    await server.wait_closed()

if __name__ == "__main__":
    asyncio.run(main())