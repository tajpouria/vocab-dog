import os
import sys
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, ContextTypes, filters

load_dotenv()


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    print(f"Received message: {update.message.text}")


telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
gemini_api_key = os.getenv("GEMINI_API_KEY")

if not telegram_token or not gemini_api_key:
    print("Error: Missing required environment variables", file=sys.stderr)
    sys.exit(1)

app = ApplicationBuilder().token(telegram_token).build()

app.add_handler(MessageHandler(filters.TEXT, handle_message))

app.run_polling()
