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


class Collocation(BaseModel):
    phrase: str = Field(description="Common phrase or collocation with the word")
    meaning: str = Field(description="Meaning of the collocation")


class SynonymAntonym(BaseModel):
    word: str = Field(description="The synonym or antonym word")
    translation: str = Field(description="Translation of the synonym or antonym")


class WordDefinition(BaseModel):
    word: str = Field(description="The word being defined")
    pronunciation: Optional[str] = Field(
        default=None, description="Phonetic pronunciation or pronunciation guide"
    )
    word_forms: Optional[str] = Field(
        default=None,
        description="Different forms of the word (plural, past tense, etc.)",
    )
    part_of_speech: str = Field(
        description="Part of speech (noun, verb, adjective, etc.)"
    )
    translation: str = Field(description="Direct translation to target language")
    definition_simple: str = Field(description="Simple, beginner-friendly definition")

    synonyms: List[SynonymAntonym] = Field(default=[], description="List of synonyms with translations")
    antonyms: List[SynonymAntonym] = Field(default=[], description="List of antonyms with translations")
    examples: List[WordExample] = Field(
        default=[], description="List of practical examples"
    )
    collocations: List[Collocation] = Field(
        default=[], description="Common word combinations"
    )
    memory_tip: Optional[str] = Field(
        default=None, description="Memory aid or mnemonic to remember the word"
    )


class WordTranslation(BaseModel):
    word: str = Field(description="The original word")
    translation: str = Field(description="Translation of the word")


class ProgressiveBreakdown(BaseModel):
    fragment: str = Field(description="The progressive fragment of the sentence")
    translation: str = Field(description="Translation of this fragment")


class SentenceBreakdown(BaseModel):
    original_text: str = Field(description="The original sentence or paragraph")
    full_translation: str = Field(description="Complete translation of the text")
    word_by_word: List[WordTranslation] = Field(description="Word-by-word breakdown with translations")
    progressive_breakdown: List[ProgressiveBreakdown] = Field(description="Progressive sentence building breakdown")


async def generate_word_audio(text: str) -> Optional[str]:
    try:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
        temp_path = temp_file.name
        temp_file.close()
        
        translator = Translator()
        async with aiofiles.open(temp_path, "wb") as file:
            await translator.tts(text, file=file, targetlang='nl')
        
        return temp_path
    except Exception as e:
        logger.error(f"Audio generation failed for '{text}': {e}")
        return None


async def get_word_definition(word: str, source_language: str, target_language: str) -> WordDefinition:
    prompt = f"""
    You are a friendly language teacher helping a beginner learner understand the word "{word}" in {source_language}.
    
    Provide a comprehensive but accessible definition that includes:
    
    1. The word in {source_language}
    2. Pronunciation guide (phonetic or simple pronunciation)
    3. Different word forms (plurals, verb conjugations, etc.) if applicable
    4. Part of speech clearly explained
    5. Direct translation to {target_language}
    6. Simple definition that a beginner can understand

    9. Difficulty level assessment
    10. 3-5 synonyms (simpler alternatives for beginners) with their translations
    11. 2-3 antonyms if applicable with their translations
    12. 5 realistic examples from different contexts:
        - Daily conversation
        - Social media/texting
        - Workplace/school
        - News/media
        - Books/stories
    13. 3-5 common collocations (word combinations)
    14. A memory tip or mnemonic to help remember the word
    
    Make everything natural, conversational, and appropriate for a beginner level learner.
    Use real contexts and situations they might encounter.
    Avoid overly academic or complex explanations.
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


async def get_sentence_breakdown(text: str, source_language: str, target_language: str) -> SentenceBreakdown:
    prompt = f"""
    You are a language teacher helping a student understand this {source_language} text: "{text}"
    
    Provide:
    1. The original text exactly as provided
    2. A complete, natural translation to {target_language}
    3. Word-by-word breakdown with individual translations (exclude punctuation, include only meaningful words)
    4. Progressive breakdown showing how the sentence builds up piece by piece, starting with the first meaningful fragment and adding more words/phrases progressively until the complete sentence is formed
    
    For the progressive breakdown, show how each fragment grows and how the meaning develops. Start with the smallest meaningful unit and build up logically.
    
    Keep translations simple and contextually appropriate.
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
