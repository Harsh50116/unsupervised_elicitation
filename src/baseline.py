import random
from setup import load_truthfulqa, setup_hyperbolic_client, format_truthfulqa_example

random.seed(76)

# Zero shot evaluations
def zero_shot_eval(client, test_data, model):
    correct = 0

    for example in test_data:
        print("-", end=" ", flush=True) 
        prompt = format_truthfulqa_example(example['question'], example['choice'])
        response = client.completions.create(
            model=model,
            prompt=prompt,
            max_tokens=1,
            temperature=0,
            logprobs=5
        )

        prediction = response.choices[0].text.strip().lower()
        prediction_label = 1 if 'true' in prediction else 0 

        if prediction_label == example['label']:
            correct += 1
        
    accuracy = correct / len(test_data) * 100
    return accuracy


# Few shot evaluations
def few_shot_eval(client, train_data, test_data, n_examples, model):
    context_examples = random.sample(train_data, min(n_examples, len(train_data)))

    context = "" 
    for ex in context_examples:
        context += format_truthfulqa_example(ex['question'], ex['choice'], ex['label'])
        context += "\n"

    # for c in context_examples:
    #     print(c)
    #     print("-" * 100)
    
    correct = 0

    for example in test_data:
        print("-", end=" ", flush=True)
        prompt = context + format_truthfulqa_example(example['question'], example['choice'])
        response = client.completions.create(
            model=model,
            prompt=prompt,
            max_tokens=1,
            temperature=0,
            logprobs=5
        )

        prediction = response.choices[0].text.strip().lower()
        prediction_label = 1 if 'true' in prediction else 0

        if prediction_label == example['label']:
            correct += 1 
        
    accuracy = correct / len(test_data) * 100
    return accuracy



def main():
    # Load train and test data 
    train_data = load_truthfulqa('../data/truthfulqa_train.json')
    test_data = load_truthfulqa('../data/truthfulqa_test.json')
    icm_labels = load_truthfulqa('../data/icm_labels.json')

    print(f"Train: {len(train_data)}, Test: {len(test_data)}")

    client = setup_hyperbolic_client()
    model = "meta-llama/Meta-Llama-3.1-405B"

    print("\nRunning baselines")

    print("\n Zero shot...")
    accuracy_zero_shot = zero_shot_eval(client, test_data, model)
    print(f"Accurancy for zero-shot use case: {accuracy_zero_shot}")

    print("\n Few shot...")
    accuracy_few_shot = few_shot_eval(client, train_data, test_data, n_examples=25, model=model)
    print(f"Accuracy for few-shot use case: {accuracy_few_shot}")
    
    print("\n ICM...")
    accuracy_icm = few_shot_eval(client, icm_labels, test_data, n_examples=25, model=model)
    print(f"Accuracy for few-shot use case: {accuracy_icm}")

if __name__ == '__main__':
    main()