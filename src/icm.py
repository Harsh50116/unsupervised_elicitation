import random
import math
import json
from setup import load_truthfulqa, setup_hyperbolic_client, format_truthfulqa_example



def calculate_mutual_predictability(client, labeled_data, target_indices=None, model="meta-llama/Meta-Llama-3.1-405B", context_size=10, sample_size=8):
    """ Calculate the log probabilities of the labeled data """
    total_score = 0.0 

    if target_indices is None:
        target_indices = random.sample(
            range(len(labeled_data)),
            min(sample_size, len(labeled_data))
        )

    for target_idx in target_indices:
        if target_idx >= len(labeled_data):
            continue
            
        target = labeled_data[target_idx]
        
        # Index-based exclusion
        other_indices = [i for i in range(len(labeled_data)) if i != target_idx]
        # deterministic randomness so old vs new are comparable
        rng = random.Random(target_idx)  # or pass a seed from run_icm if you prefer
        context_indices = rng.sample(other_indices, min(context_size, len(other_indices)))

        context_examples = [labeled_data[i] for i in context_indices]

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

    m = len(target_indices)
    return (total_score / m) if m > 0 else 0.0 


def calculate_inconsistency(labeled_data):
    """Calculate I(D): penalty if all labels same"""
    labels = [ex['label'] for ex in labeled_data]
    all_same = len(set(labels)) == 1
    return 10 if all_same else 0


def calculate_score(client, labeled_data, target_indices, alpha=50, context_size=10):
    """formula:
        U(D) = alpha * P_B(D) - I(D)  
    """
    mutual = calculate_mutual_predictability(client, labeled_data, target_indices=target_indices, context_size=context_size)
    inconsist = calculate_inconsistency(labeled_data)
    return alpha * mutual - inconsist 


def propose_label(client, example, context_examples, model="meta-llama/Meta-Llama-3.1-405B", context_size=10):
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
    sample_size = 8

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
        proposed_label = propose_label(client, example, context, model, context_size=10)

        # Create new dataset with proposed label 
        if is_new:
            new_labeled = labeled_data + [dict(example, label=proposed_label)]
        else:
            new_labeled = labeled_data.copy()
            new_labeled[idx] = dict(example, label=proposed_label)
        

        # Calculate scores 
        sample_size = 8

        if is_new:
            # new example sits at the end of new_labeled
            new_idx = len(new_labeled) - 1

            pop_size = min(sample_size, len(new_labeled))
            score_targets = random.sample(range(len(new_labeled)), pop_size)

            # force include the new example
            if new_idx not in score_targets:
                score_targets[0] = new_idx

            old_score = calculate_score(client, labeled_data, target_indices=score_targets, context_size=10)
            new_score = calculate_score(client, new_labeled, target_indices=score_targets, context_size=10)

        else:
            pop_size = min(sample_size, len(labeled_data))
            score_targets = random.sample(range(len(labeled_data)), pop_size)

            # force include idx (the relabeled example)
            if idx not in score_targets:
                score_targets[0] = idx

            old_score = calculate_score(client, labeled_data, target_indices=score_targets, context_size=10)
            new_score = calculate_score(client, new_labeled, target_indices=score_targets, context_size=10)

        delta = new_score - old_score


        # Normal accept/reject
        if delta > 0 or random.random() < math.exp(delta / T):
            labeled_data = new_labeled
            if is_new:
                unlabeled_indices.remove(idx)
            print(f"Iter {iteration}: Accepted | Score: {new_score:.2f} | Labeled: {len(labeled_data)}")
        else:
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
        max_iterations=256,
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




