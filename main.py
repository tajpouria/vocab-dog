import os
import sys
import signal
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, ContextTypes, filters
from jinja2 import Template

load_dotenv()

from shared import get_logger
from word_service import get_word_definition, generate_word_audio

logger = get_logger(__name__)

# Message formatting template
MESSAGE_TEMPLATE = Template("""
<b>{{ word }}</b>{% if word_forms %} <i>({{ word_forms }})</i>{% endif %} <em>[{{ part_of_speech }}]</em>
{% if pronunciation %}<b><i>{{ pronunciation }}</i></b>{% endif %}
<u>{{ translation }}</u>

<em>{{ definition_simple }}</em>

{% if synonyms %}<b>Synonyms:</b>
{% for synonym in synonyms %}- <em>{{ synonym.word }}</em> <u>({{ synonym.translation }})</u>{% if not loop.last %}, {% endif %}
{% endfor %}{% endif %}
{% if antonyms %}<b>Antonyms:</b>
{% for antonym in antonyms %}- <em>{{ antonym.word }}</em> <u>{({ antonym.translation }})</u>{% if not loop.last %}, {% endif %}
{% endfor %}{% endif %}
{% if examples %}<b>Examples:</b>
{% for example in examples %}- <i>"{{ example.example }}"</i>
   <u>{{ example.translation }}</u>
{% endfor %}{% endif %}
{% if collocations %}<b>Collocations:</b>
{% for collocation in collocations %}- {{ collocation.phrase }} <u>({{ collocation.meaning }})</u>
{% endfor %}{% endif %}
{% if memory_tip %}<b>Tip:</b> <i>{{ memory_tip }}</i>{% endif %}
""")


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

    word = update.message.text.strip()
    logger.info(f"Received message: {word}")

    if not word or len(word.split()) != 1:
        await update.message.reply_text(
            f"Please send a single word. I'll translate from {source_language} to {target_language}."
        )
        return

    if update.effective_chat:
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    try:
        definition = await get_word_definition(word, source_language, target_language)
        audio_path = await generate_word_audio(word)
        
        message = MESSAGE_TEMPLATE.render(**definition.model_dump()).strip()
        
        if audio_path:
            try:
                with open(audio_path, "rb") as audio_file:
                    await update.message.reply_voice(
                        voice=audio_file,
                        caption=f"Dutch pronunciation of '{word}'"
                    )
                await update.message.reply_text(message, parse_mode="HTML")
                os.unlink(audio_path)
            except Exception as e:
                logger.error(f"Audio send failed: {e}")
                if os.path.exists(audio_path):
                    os.unlink(audio_path)
                await update.message.reply_text(message, parse_mode="HTML")
        else:
            await update.message.reply_text(message, parse_mode="HTML")
            
    except Exception as e:
        logger.error(f"Definition failed for '{word}': {e}")
        await update.message.reply_text(f"Sorry, couldn't define '{word}'. Try again.")


def main():
    app = ApplicationBuilder().token(telegram_token).build()
    app.add_handler(MessageHandler(filters.TEXT, handle_message))
    
    def signal_handler(signum, frame):
        logger.info("Shutting down...")
        app.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
