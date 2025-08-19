import os
import sys
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes


async def hello(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(f"Hello {update.effective_user.first_name}")


telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
gemini_api_key = os.getenv("GEMINI_API_KEY")

if not telegram_token or not gemini_api_key:
    print("Error: Missing required environment variables", file=sys.stderr)
    sys.exit(1)

app = ApplicationBuilder().token(telegram_token).build()

app.add_handler(CommandHandler("hello", hello))

app.run_polling()
