import json

import requests

model: str = "CodeLlama:7b"

url: str = "http://localhost:11434/api/generate"

data: dict[str, str] = {
    "model": model,
    "prompt": "tell me a short story and make it funny.",
}

response: requests.Response = requests.post(
    url=url, json=data, stream=True
)  # remove the stream=True to get the full response


# check the response status
if response.status_code == 200:
    print("Generated Text:", end=" ", flush=True)
    # Iterate over the streaming response
    for line in response.iter_lines():
        if line:
            # Decode the line and parse the JSON
            decoded_line: str = line.decode("utf-8")
            result: dict[str, str] = json.loads(s=decoded_line)
            # Get the text from the response
            generated_text = result.get("response", "")
            print(generated_text, end="", flush=True)
else:
    print("Error:", response.status_code, response.text)
