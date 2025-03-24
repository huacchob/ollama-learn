import os
from pathlib import Path

import ollama

from .utility import find_root_directory

current_dir: Path = find_root_directory(file=__file__)

model: str = "CodeLlama:7b"

# Paths to input and output files
input_file: Path = current_dir.joinpath("data/grocery_list.txt")
output_file: Path = current_dir.joinpath("data/categorized_grocery_list.txt")


# Check if the input file exists
if not os.path.exists(path=input_file):
    print(f"Input file '{input_file}' not found.")
    exit(code=1)


# Read the uncategorized grocery items from the input file
with open(file=input_file, mode="r", encoding="utf-8") as f:
    items: str = f.read().strip()


# Prepare the prompt for the model
prompt: str = f"""
You are an assistant that categorizes and sorts grocery items.

Here is a list of grocery items:

{items}

Please:

1. Categorize these items into appropriate categories such as Produce, Dairy, Meat, Bakery, Beverages, etc.
2. Sort the items alphabetically within each category.
3. Present the categorized list in a clear and organized manner, using bullet points or numbering.

"""


# Send the prompt and get the response
try:
    response: ollama.GenerateResponse = ollama.generate(
        model=model,
        prompt=prompt,
    )
    response.context
    generated_text: str = response.get(key="response")
    if generated_text:
        print("==== Categorized List: ===== \n")
        print(generated_text)

        # Write the categorized list to the output file
        with open(file=output_file, mode="w") as f:
            f.write(generated_text.strip())

        print(f"Categorized grocery list has been saved to '{output_file}'.")
except Exception as e:
    print("An error occurred:", str(object=e))
