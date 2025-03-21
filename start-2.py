import ollama

model: str = "CodeLlama:7b"

response: ollama.ListResponse = ollama.list()

# print(response)

# == Chat example ==
res: ollama.ChatResponse = ollama.chat(
    model=model,
    messages=[
        {"role": "user", "content": "why is the sky blue?"},
    ],
)
# print(res["message"]["content"])

# == Chat example streaming ==
res: ollama.ChatResponse = ollama.chat(
    model=model,
    messages=[
        {
            "role": "user",
            "content": "why is the ocean so salty?",
        },
    ],
    stream=True,
)
# for chunk in res:
#     print(chunk["message"]["content"], end="", flush=True)


# ==================================================================================
# ==== The Ollama Python library's API is designed around the Ollama REST API ====
# ==================================================================================

# == Generate example ==
res: ollama.GenerateResponse = ollama.generate(
    model=model,
    prompt="why is the sky blue?",
)

# show
# print(ollama.show(model))


# Create a new model with modelfile
new_model: str = "knowitall"

modelfile: str = """
FROM llama3.2
SYSTEM You are very smart assistant who knows everything about oceans. You are very succinct and informative.
PARAMETER temperature 0.1
"""

ollama.create(model=new_model, modelfile=modelfile)

res: ollama.GenerateResponse = ollama.generate(
    model=new_model,
    prompt="why is the ocean so salty?",
)
print(res["response"])


# delete model
ollama.delete(model=new_model)
