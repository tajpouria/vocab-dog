from pydantic import BaseModel, Field
from typing import Optional, List
from shared import get_logger, genai_client, genai_model
import tempfile
import aiofiles
from gpytranslate import Translator

logger = get_logger(__name__)


class WordExample(BaseModel):
    example: str = Field(description="The example sentence using the word")
    translation: str = Field(description="Translation of the example")


class SynonymAntonym(BaseModel):
    word: str = Field(description="The synonym or antonym word")
    translation: str = Field(description="Translation of the synonym or antonym")


class WordDefinition(BaseModel):
    word: str = Field(description="The word being defined")
    word_forms: Optional[str] = Field(
        default=None,
        description="Different forms of the word (plural, past tense, etc.)",
    )
    translation: str = Field(description="Direct translation to target language")
    definition_simple: str = Field(description="Simple, beginner-friendly definition")

    synonyms: List[SynonymAntonym] = Field(default=[], description="List of synonyms with translations")
    antonyms: List[SynonymAntonym] = Field(default=[], description="List of antonyms with translations")
    examples: List[WordExample] = Field(
        default=[], description="List of practical examples"
    )


class WordTranslation(BaseModel):
    word: str = Field(description="The original word")
    translation: str = Field(description="Translation of the word")


class FragmentBreakdown(BaseModel):
    fragment: str = Field(description="A meaningful fragment of the sentence")
    translation: str = Field(description="Translation of this fragment")


class SentenceBreakdown(BaseModel):
    original_text: str = Field(description="The original sentence or paragraph")
    full_translation: str = Field(description="Complete translation of the text")
    word_by_word: List[WordTranslation] = Field(
        description="Word-by-word breakdown with translations"
    )
    fragment_breakdown: List[FragmentBreakdown] = Field(
        description="Sentence broken down into meaningful fragments with translations"
    )


async def generate_word_audio(text: str) -> Optional[str]:
    try:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        temp_path = temp_file.name
        temp_file.close()

        translator = Translator()
        async with aiofiles.open(temp_path, "wb") as file:
            await translator.tts(text, file=file, targetlang="nl")

        return temp_path
    except Exception as e:
        logger.error(f"Audio generation failed for '{text}': {e}")
        return None


async def get_word_definition(
    word: str, source_language: str, target_language: str
) -> WordDefinition:
    prompt = f"""
    You are a friendly language teacher helping a beginner learner understand the word "{word}" in {source_language}.
    
    Provide an accessible definition that includes:
    
    - The word in {source_language}
    - Different word forms (plurals, verb conjugations, etc.) if applicable
    - Direct translation to {target_language}
    - Simple definition that a beginner can understand
    - 3 synonyms with their translations
    - 3 realistic examples from different contexts
    
    Make everything natural, conversational, and appropriate for a beginner level learner.
    Use real contexts and situations they might encounter.
    """

    response = genai_client.models.generate_content(
        model=genai_model,
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": WordDefinition,
            "temperature": 0.1,
        },
    )

    if not response.parsed:
        raise ValueError("No response from API")

    return response.parsed


async def get_sentence_breakdown(
    text: str, source_language: str, target_language: str
) -> SentenceBreakdown:
    prompt = f"""
    You are a language teacher helping a student understand this {source_language} text: "{text}"
    
    The original text exactly as provided
    A complete, natural translation to {target_language}
    Word-by-word breakdown with individual translations (exclude punctuation, include only meaningful words)
    A breakdown of the sentence into meaningful fragments, with a translation for each fragment.
    The fragments should be sequential and cover the entire sentence.
    """

    response = genai_client.models.generate_content(
        model=genai_model,
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": SentenceBreakdown,
            "temperature": 0.1,
        },
    )

    if not response.parsed:
        raise ValueError("No response from API")

    return response.parsed
