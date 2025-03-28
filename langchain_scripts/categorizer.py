"""Generate response from Ollama."""

from pathlib import Path

import ollama

from .utility import find_root_directory


class OllamaInterface:
    """Interface with Ollama."""

    def __init__(self) -> None:
        """Initialize."""

    def generate_response(
        self,
        model: str,
        prompt: str,
    ) -> str:
        """Generate a response from Ollama.

        Args:
            model (str): Model to use.
            prompt (str): Prompt to send to the model.

        Returns:
            str: Generated response.
        """
        response: ollama.GenerateResponse = ollama.generate(
            model=model,
            prompt=prompt,
        )
        return response.get(key="response")


current_dir: Path = find_root_directory(file=__file__)

# Paths to input and output files
input_file: Path = current_dir.joinpath("data/grocery_list.txt")
output_file: Path = current_dir.joinpath("data/categorized_grocery_list.txt")


# Read the un-categorized grocery items from the input file
with open(file=input_file, mode="r", encoding="utf-8") as f:
    items: str = f.read().strip()

ollama_model: str = "CodeLlama:7b"

ollama_prompt: str = f"""
You are an assistant that categorizes and sorts grocery items.

Here is a list of grocery items:

{items}

Please:

1. Categorize these items into appropriate categories such as Produce, Dairy,
    Meat, Bakery, Beverages, etc.
2. Sort the items alphabetically within each category.
3. Present the categorized list in a clear and organized manner, using bullet
    points or numbering.

"""

# Send the prompt and get the response
ollama_interface: OllamaInterface = OllamaInterface()
generated_text: str = ollama_interface.generate_response(
    model=ollama_model,
    prompt=ollama_prompt,
)
if generated_text:
    print(generated_text)

    # Write the categorized list to the output file
    with open(file=output_file, mode="w", encoding="utf-8") as f:
        f.write(generated_text.strip())

    print(f"Ollama response has been saved to '{output_file}'.")
