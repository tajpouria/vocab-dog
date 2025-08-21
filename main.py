import os
import sys
import signal
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, ContextTypes, filters
from jinja2 import Template

load_dotenv()

from shared import get_logger
from word_service import get_word_definition, generate_word_audio, get_sentence_breakdown

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
{% for antonym in antonyms %}- <em>{{ antonym.word }}</em> <u>{({{ antonym.translation }})</u>{% if not loop.last %}, {% endif %}
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

# Sentence formatting template
SENTENCE_TEMPLATE = Template("""
<b>{{ original_text }}</b>

<u>{{ full_translation }}</u>

{% for breakdown in progressive_breakdown %}"{{ breakdown.fragment }}"
<u>{{ breakdown.translation }}</u>

{% endfor %}
{% for word_trans in word_by_word %}<em>{{ word_trans.word }}</em> (<u>{{ word_trans.translation }}</u>)
{% endfor %}
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

    text = update.message.text.strip()
    logger.info(f"Received message: {text}")

    if not text:
        return

    if update.effective_chat:
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    try:
        # Check if it's a single word or sentence/paragraph
        if len(text.split()) == 1:
            # Single word - use existing word definition logic
            definition = await get_word_definition(text, source_language, target_language)
            audio_path = await generate_word_audio(text)
            
            message = MESSAGE_TEMPLATE.render(**definition.model_dump()).strip()
            
            if audio_path:
                try:
                    with open(audio_path, "rb") as audio_file:
                        await update.message.reply_voice(
                            voice=audio_file,
                            caption=f"{source_language.title()} pronunciation of '{text}'"
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
        else:
            # Sentence/paragraph - use sentence breakdown logic
            breakdown = await get_sentence_breakdown(text, source_language, target_language)
            audio_path = await generate_word_audio(text)
            
            message = SENTENCE_TEMPLATE.render(**breakdown.model_dump()).strip()
            
            if audio_path:
                try:
                    with open(audio_path, "rb") as audio_file:
                        await update.message.reply_voice(
                            voice=audio_file,
                            caption=f"{source_language.title()} pronunciation"
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
        logger.error(f"Processing failed for '{text}': {e}")
        await update.message.reply_text(f"Sorry, couldn't process '{text}'. Try again.")


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
