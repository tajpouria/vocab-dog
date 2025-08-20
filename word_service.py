from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from jinja2 import Template
from shared import get_logger, genai_client, genai_model

logger = get_logger(__name__)


MESSAGE_TEMPLATE = Template(
    """
<b>{{ word_header }}</b>{% if word_forms %} <i>({{ word_forms }})</i>{% endif %} <em>[{{ part_of_speech }}]</em>
{% if pronunciation -%}
<b><i>{{ pronunciation }}</i></b>
{% endif %}
<u>{{ translation }}</u>

<em>{{ definition_simple }}</em>

{% if synonyms -%}
<b>Synonyms:</b>

{% for synonym in synonyms -%}
- <em>{{ synonym.word }}</em> - <u>{{ synonym.translation }}</u>{% if not loop.last %}, {% endif %}
{% endfor %}
{% endif -%}
{% if antonyms -%}
<b>Antonyms:</b>

{% for antonym in antonyms -%}
- <em>{{ antonym.word }}</em> - <u>{{ antonym.translation }}</u>{% if not loop.last %}, {% endif %}
{% endfor %}
{% endif -%}
{% if examples -%}
<b>Examples:</b>

{% for example in examples -%}
- <i>"{{ example.example }}"</i>
   <em>{{ example.translation }}</em>
{% endfor -%}
{% endif %}
{% if collocations -%}
<b>Collocations:</b>

{% for collocation in collocations -%}
- {{ collocation.phrase }} <u>({{ collocation.meaning }})</u>
{% endfor %}
{% endif -%}
{% if memory_tip -%}
<b>💡 Tip:</b> <i>{{ memory_tip }}</i>
{% endif -%}
"""
)


class WordExample(BaseModel):
    example: str = Field(description="The example sentence using the word")
    translation: str = Field(description="Translation of the example")


class Collocation(BaseModel):
    phrase: str = Field(description="Common phrase or collocation with the word")
    meaning: str = Field(description="Meaning of the collocation")


class SynonymAntonym(BaseModel):
    word: str = Field(description="The synonym or antonym word")
    translation: str = Field(description="Translation of the synonym or antonym")


class EnhancedWordDefinition(BaseModel):
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


async def generate_enhanced_word_definition(
    word: str, source_language: str, target_language: str, user_level: str = "beginner"
) -> Dict:
    """
    Generate a comprehensive, beginner-friendly word definition.

    Args:
        word: The word to define
        source_language: Language of the word
        target_language: Language to translate to
        user_level: User's language level (beginner, intermediate, advanced)

    Returns:
        Dict containing both formatted message and raw data
    """
    logger.info(f"Generating definition for word: '{word}' ({source_language} -> {target_language})")

    prompt = f"""
    You are a friendly language teacher helping a {user_level} learner understand the word "{word}" in {source_language}.
    
    Provide a comprehensive but accessible definition that includes:
    
    1. The word in {source_language}
    2. Pronunciation guide (phonetic or simple pronunciation)
    3. Different word forms (plurals, verb conjugations, etc.) if applicable
    4. Part of speech clearly explained
    5. Direct translation to {target_language}
    6. Simple definition that a {user_level} can understand

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
    
    Make everything natural, conversational, and appropriate for a {user_level} level learner.
    Use real contexts and situations they might encounter.
    Avoid overly academic or complex explanations.
    """

    try:
        logger.debug(f"Calling Gemini API with model: {genai_model}")
        response = genai_client.models.generate_content(
            model=genai_model,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": EnhancedWordDefinition,
                "temperature": 0.1,
            },
        )

        if not response.parsed:
            raise ValueError("No parsed response received from Gemini API")

        definition = response.parsed
        if not isinstance(definition, EnhancedWordDefinition):
            raise ValueError("Invalid response format from Gemini API")

        formatted_message = MESSAGE_TEMPLATE.render(
            word_header=f"{definition.word}",
            pronunciation=definition.pronunciation,
            word_forms=definition.word_forms,
            part_of_speech=definition.part_of_speech,
            translation=definition.translation,
            definition_simple=definition.definition_simple,

            synonyms=definition.synonyms,
            antonyms=definition.antonyms,
            examples=definition.examples,
            collocations=definition.collocations,
            memory_tip=definition.memory_tip,
        ).strip()

        logger.info(f"Successfully generated definition for '{word}' with {len(definition.examples)} examples and {len(definition.synonyms)} synonyms")
        return {
            "success": True,
            "formatted_message": formatted_message,
            "raw_data": definition.model_dump(),
            "word": definition.word,
            "translation": definition.translation,
            "examples_count": len(definition.examples),
            "synonyms_count": len(definition.synonyms),
        }

    except Exception as e:
        logger.error(f"Failed to generate definition for '{word}': {str(e)}")
        error_message = (
            f"Sorry, I couldn't generate a definition for '{word}'. Please try again."
        )
        return {
            "success": False,
            "error": str(e),
            "formatted_message": error_message,
            "raw_data": None,
        }


async def generate_word_definition(
    word: str, source_language: str, target_language: str
) -> str:
    """
    Backward compatibility wrapper for the enhanced function.
    Returns only the formatted message string as before.
    """
    logger.debug(f"Wrapper function called for word: '{word}'")
    result = await generate_enhanced_word_definition(
        word, source_language, target_language
    )
    return result["formatted_message"]
