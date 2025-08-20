import os
import sys
import signal
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, ContextTypes, filters

load_dotenv()

from shared import get_logger
from word_service import generate_word_definition

logger = get_logger(__name__)


telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
gemini_api_key = os.getenv("GEMINI_API_KEY")
source_language = os.getenv("SOURCE_LANGUAGE", "dutch")
target_language = os.getenv("TARGET_LANGUAGE", "english")

if not telegram_token or not gemini_api_key:
    logger.error("Missing required environment variables")
    sys.exit(1)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text:
        return

    message_text = update.message.text.strip()
    logger.info(f"Received message: {message_text}")

    word = message_text

    if not word or len(word.split()) != 1:
        await update.message.reply_text(
            f"Please send a single word to get its definition.\n"
            f"I'll translate from {source_language} to {target_language}.\n"
            f"Example: pull"
        )
        return

    if update.effective_chat and update.effective_chat.id:
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id, action="typing"
        )

    definition = await generate_word_definition(word, source_language, target_language)

    await update.message.reply_text(definition, parse_mode="HTML")


app = ApplicationBuilder().token(telegram_token).build()

app.add_handler(MessageHandler(filters.TEXT, handle_message))


def signal_handler(signum, frame):
    logger.info("Shutting down gracefully...")
    app.stop()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

app.run_polling(drop_pending_updates=True)
