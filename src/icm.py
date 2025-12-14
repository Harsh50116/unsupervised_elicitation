import random
import math
import json
from setup import load_truthfulqa, setup_hyperbolic_client, format_truthfulqa_example



def calculate_mutual_predictability(client, labeled_data, model="meta-llama/Meta-Llama-3.1-405B", context_size=20):
    """ Calculate the log probabilities of the labeled data """
    total_score = 0.0 

    for i, target in enumerate(labeled_data):

        other_examples = [ex for j, ex in enumerate(labeled_data) if i!=j] 
        context_examples = random.sample(other_examples, min(context_size, len(other_examples)))

        context = ""
        for ex in context_examples:
            context += format_truthfulqa_example(ex['question'], ex['choice'], ex['label'])
            context += "\n"
        
        # Prompting without answer 
        prompt = context + format_truthfulqa_example(target['question'], target['choice'])

        response = client.completions.create(
            model=model, 
            prompt=prompt, 
            max_tokens=1,
            temperature=0,
            logprobs=5
        )

        #  Get logprob for actual label 
        top_logprobs = response.choices[0].logprobs.top_logprobs[0]
        actual_label = "True" if target['label'] == 1 else 'False'

        label_logprob = -10.0  
        for token, logprob in top_logprobs.items():
            # print(token + " : " + str(logprob))
            if actual_label.lower() in token.lower():
                label_logprob = logprob 
                break 

        total_score += label_logprob
    
    return total_score 


def calculate_inconsistency(labeled_data):
    """Calculate I(D): penalty if all labels same"""
    labels = [ex['label'] for ex in labeled_data]
    all_same = len(set(labels)) == 1
    return 10 if all_same else 0


def calculate_score(client, labeled_data, alpha=50, context_size=20):
    """formula:
        U(D) = alpha * P_B(D) - I(D)  
    """
    mutual = calculate_mutual_predictability(client, labeled_data, context_size=context_size)
    inconsist = calculate_inconsistency(labeled_data)
    return alpha * mutual - inconsist 


def propose_label(client, example, context_examples, model="meta-llama/Meta-Llama-3.1-405B", context_size=20):
    """
    Args: 
        client: The client to use to make API calls.
        example: The example to label.
        context_examples: The context examples to use to label the example.
        model: The model to use to make API calls.
        context_size: The number of context examples to use to label the example.
    """

    sampled_context = random.sample(context_examples, min(context_size, len(context_examples)))

    context = ""
    for ex in sampled_context:
        context += format_truthfulqa_example(ex['question'], ex['choice'], ex['label'])
        context += "\n"

    prompt = context + format_truthfulqa_example(example['question'], example['choice'])

    response = client.completions.create(
        model=model,
        prompt=prompt,
        max_tokens=1,
        temperature=0,
        logprobs=5
    )

    # Pick label with highes probability 
    top_logprobs = response.choices[0].logprobs.top_logprobs[0] 

    # print(response.choices[0].logprobs)

    # print("-" * 100)

    best_label = None 
    best_logprob = -float('inf')

    for token, logprob in top_logprobs.items():
        # print(token + " : " + str(logprob))
        if 'true' in token.lower():
            if logprob > best_logprob:
                best_label = 1
                best_logprob= logprob 
        elif 'false' in token.lower():
            if logprob > best_logprob:
                best_label = 0
                best_logprob = logprob 
    
    # print("-" * 100)
    
    if best_label is None:
        best_label = random.choice([0, 1])
    
    return best_label





def run_icm(client, unlabeled_data, K=8, max_iterations=256, model="meta-llama/Meta-Llama-3.1-405B"):
    """
    Args:
        client: The client to use to make API calls.
        unlabeled_data: The unlabeled data to label.
        K: The number of closest neighbors to consider.
        max_iterations: The maximum number of iterations to run.
        model: The model to use to make API calls.
    """

    labeled_data = []
    random.shuffle(unlabeled_data)

    for i in range(K):
        example = unlabeled_data[i].copy() 

        example['label'] = random.choice([0, 1])
        labeled_data.append(example)
    
    print(f"Initialized with {K} labeled exmaples")

    # Temperature parameters 
    T0 = 10.0 
    Tmin = 0.01
    beta = 0.99 

    # Track unlabeled 
    unlabeled_indices = set(range(K, len(unlabeled_data)))

    for iteration in range(max_iterations):
        # Update temperature 
        T = max(Tmin, T0 / (1 + beta * math.log(iteration + 1)))

        # Sample example to label 
        if unlabeled_indices and random.random() < 0.8:
            # Prioritize unlabeled 
            idx = random.choice(list(unlabeled_indices))
            example = unlabeled_data[idx].copy() 
            is_new = True 
        else:
            # Relabel existing 
            idx = random.randint(0, len(labeled_data) - 1)
            example = labeled_data[idx].copy() 
            is_new = False 
        
        # Propose label 
        context = [ex for ex in labeled_data if ex != example]
        proposed_label = propose_label(client, example, context, model, context_size=20)

        # Create new dataset with proposed label 
        if is_new:
            new_labeled = labeled_data + [dict(example, label=proposed_label)]
        else:
            new_labeled = labeled_data.copy()
            new_labeled[idx] = dict(example, label=proposed_label)
        

        # Calculate scores 
        old_score = calculate_score(client, labeled_data, context_size=20) 
        new_score = calculate_score(client, new_labeled, context_size=20)
        delta = new_score - old_score 

        if delta > 0 or random.random() < math.exp(delta / T):
            labeled_data = new_labeled
            if is_new:
                unlabeled_indices.remove(idx)
            
            # if iteration % 20 == 0:
            print(f"Iter {iteration}: Accepted | Score: {new_score:.2f} | Labeled: {len(labeled_data)}")
        else:
            # if iteration % 20 == 0:
            print(f"Iter {iteration}: Rejected | Score: {old_score:.2f} | Labeled: {len(labeled_data)}")
    
    print(f"\nICM complete: {len(labeled_data)}/{len(unlabeled_data)} labeled")
    return labeled_data



def save_labels(labeled_data, filepath):
    with open(filepath, 'w') as f:
        json.dump(labeled_data, f, indent=2)
    print(f"Labeld saved to {filepath}")


def main():
    train_data = load_truthfulqa('../data/truthfulqa_train.json')
    client = setup_hyperbolic_client()

    print("Runing ICM Algorithm")
    print(f"Dataset size: {len(train_data)}")

    labeled_data = run_icm(
        client,
        train_data,
        K=8,
        max_iterations=1,
        model="meta-llama/Meta-Llama-3.1-405B"
        )

    # Save labeled data 
    save_labels(labeled_data, '../data/icm_labels.json')

    # show label distribution 
    true_count = sum(1 for ex in labeled_data if ex['label'] == 1)
    false_count = len(labeled_data) - true_count
    print(f"Label distribution: True: {true_count}, False: {false_count}")


if __name__ == '__main__':
    main()




