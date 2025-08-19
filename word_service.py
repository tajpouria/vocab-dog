import os
from google import genai
from pydantic import BaseModel
from typing import Optional, List
from jinja2 import Template

client = genai.Client()
model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# Telegram message template
MESSAGE_TEMPLATE = Template(
    """
<b>{{ word_header }}</b>

{% if word_forms -%}
[{{ word_forms }}]

{% endif -%}
{{ part_of_speech }}

<i>{{ translation }}</i>

📖 <b>Definition:</b>
{{ definition_in_source }}

🔄 <b>Translation:</b>
{{ definition_translation }}

{% if synonyms -%}
🔗 <b>Synonyms:</b> {{ synonyms | join(', ') }}

{% endif -%}
{% if examples -%}
📚 <b>Examples:</b>
{% for example in examples -%}
{{ loop.index }}. {{ example.example }}
   <i>→ {{ example.translation }}</i>

{% endfor -%}
{% endif -%}
"""
)


class WordDefintionExample(BaseModel):
    example: str
    translation: str


class WordDefinition(BaseModel):
    word: str
    word_forms: Optional[str] = None
    part_of_speech: str
    translation: str
    definition_in_source: str
    definition_translation: str
    synonyms: List[str] = []
    examples: List[WordDefintionExample] = []


async def generate_word_definition(
    word: str, source_language: str, target_language: str
) -> str:
    prompt = f"""
    Provide a comprehensive definition for the word "{word}" in {source_language}. 
    Format the response with the following information:
    
    1. The word and its language
    2. Different word forms if applicable (past tense, present participle, etc.)
    3. Part of speech (noun, verb, adjective, etc.)
    4. Translation to {target_language}
    5. Definition in {source_language}
    6. Translation of the definition to {target_language}
    7. Up to 5 synonyms if available
    8. 5 short examples from real contexts (textbooks, movies, references) with translations to {target_language}
    
    Make sure to be accurate and provide real, contextual examples.
    """

    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": WordDefinition,
                "temperature": 0.0,
            },
        )

        if not response.parsed:
            raise ValueError("No parsed response received from Gemini API")

        definition = response.parsed
        if not isinstance(definition, WordDefinition):
            raise ValueError("Invalid response format from Gemini API")

        formatted_response = MESSAGE_TEMPLATE.render(
            word_header=f"{definition.word} in {source_language}",
            word_forms=definition.word_forms,
            part_of_speech=definition.part_of_speech,
            translation=definition.translation,
            definition_in_source=definition.definition_in_source,
            definition_translation=definition.definition_translation,
            synonyms=definition.synonyms,
            examples=definition.examples,
        ).strip()

        return formatted_response

    except Exception as e:
        return f"Sorry, I couldn't generate a definition for '{word}'. Error: {str(e)}"
