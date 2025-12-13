import json
import os
from openai import OpenAI

# Load TruthfulQA dataset

def load_truthfulqa(filepath='../data/truthfulqa_train.json'):
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data



# Hyperbolic setup

def setup_hyperbolic_client():
    api_key = os.environ.get("HYPERBOLIC_API_KEY")

    if not api_key:
        raise Exception("Set hyperbolic key")

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.hyperbolic.xyz/v1"
    )

    return client



# Format example

def format_truthfulqa_example(question, choice, label=None):
    formatted = f"Question: {question}\n"
    formatted += f"Claim: {choice}\n"
    
    if label is not None:
        formatted += f"Answer: {'True' if label == 1 else 'False'}\n"
    else:
        formatted += "Answer:"
    
    return formatted



# Test setup

def test_setup(client, data):
    example = data[0]
    prompt = format_truthfulqa_example(example['question'], example['choice'])

    response = client.completions.create(
        model="meta-llama/Meta-Llama-3.1-405B",
        prompt=prompt,
        max_tokens=1,
        temperature=0,
        logprobs=5
    )

    print(response)

    # Checking logprobs
    logprobs = response.choices[0].logprobs
    if not logprobs:
        raise Exception("Log probabilities not available")
    
    print("Setup complete")
    return True 


def main():

    data = load_truthfulqa()
    client = setup_hyperbolic_client()
    test_setup(client, data)

if __name__ == "__main__":
    main()

    
    
        
