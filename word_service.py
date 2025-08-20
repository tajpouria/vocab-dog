import os
from google import genai
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from jinja2 import Template
from shared import get_logger

logger = get_logger(__name__)

client = genai.Client()
model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

MESSAGE_TEMPLATE = Template(
    """
<b>{{ word_header }}</b>

{% if pronunciation -%}
<b>Pronunciation:</b> {{ pronunciation }}

{% endif -%}
{% if word_forms -%}
<b>Word Forms:</b> {{ word_forms }}

{% endif -%}
<b>Part of Speech:</b> {{ part_of_speech }}

<b>Translation:</b> <i>{{ translation }}</i>

<b>Definition:</b>
{{ definition_simple }}

{% if definition_detailed -%}
<b>Extended Definition:</b>
{{ definition_detailed }}

{% endif -%}
{% if usage_tip -%}
<b>Usage Guidelines:</b>
{{ usage_tip }}

{% endif -%}
{% if synonyms -%}
<b>Synonyms:</b> {{ synonyms | join(', ') }}

{% endif -%}
{% if antonyms -%}
<b>Antonyms:</b> {{ antonyms | join(', ') }}

{% endif -%}
{% if difficulty_level -%}
<b>Proficiency Level:</b> {{ difficulty_level }}

{% endif -%}
{% if examples -%}
<b>Examples in Context:</b>
{% for example in examples -%}
{{ loop.index }}. <b>{{ example.context }}:</b> "{{ example.example }}"
   <i>Translation:</i> {{ example.translation }}
   {% if example.explanation -%}
   <i>Note:</i> {{ example.explanation }}
   {% endif %}

{% endfor -%}
{% endif -%}
{% if collocations -%}
<b>Common Collocations:</b>
{% for collocation in collocations -%}
• {{ collocation.phrase }} - <i>{{ collocation.meaning }}</i>
{% endfor -%}

{% endif -%}
{% if memory_tip -%}
<b>Learning Aid:</b> {{ memory_tip }}
{% endif -%}
"""
)


class WordExample(BaseModel):
    context: str = Field(
        description="Context where this example is from (e.g., 'Daily conversation', 'News', 'Literature')"
    )
    example: str = Field(description="The example sentence using the word")
    translation: str = Field(description="Translation of the example")
    explanation: Optional[str] = Field(
        default=None, description="Brief explanation of why this example is useful"
    )


class Collocation(BaseModel):
    phrase: str = Field(description="Common phrase or collocation with the word")
    meaning: str = Field(description="Meaning of the collocation")


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
    definition_detailed: Optional[str] = Field(
        default=None, description="More detailed explanation for advanced learners"
    )
    usage_tip: Optional[str] = Field(
        default=None, description="Practical tip on how to use this word"
    )
    difficulty_level: Optional[str] = Field(
        default=None, description="Difficulty level (Beginner, Intermediate, Advanced)"
    )
    synonyms: List[str] = Field(default=[], description="List of synonyms")
    antonyms: List[str] = Field(default=[], description="List of antonyms")
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
    7. More detailed definition for deeper understanding (optional)
    8. Practical usage tip explaining when/how to use this word
    9. Difficulty level assessment
    10. 3-5 synonyms (simpler alternatives for beginners)
    11. 2-3 antonyms if applicable
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
        logger.debug(f"Calling Gemini API with model: {model}")
        response = client.models.generate_content(
            model=model,
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
            definition_detailed=definition.definition_detailed,
            usage_tip=definition.usage_tip,
            difficulty_level=definition.difficulty_level,
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
            "difficulty": definition.difficulty_level,
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
