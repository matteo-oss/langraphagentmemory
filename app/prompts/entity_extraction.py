from langchain_core.prompts import ChatPromptTemplate

entity_extraction_prompt = ChatPromptTemplate.from_template(
    "Extract all entities (Person, Organization, Location) from the text below.\n"
    "Return a comma-separated list only, no explanations.\n\n"
    "Text: {text}\n\n"
    "Entities:"
)