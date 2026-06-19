Project X - Task 3: Word2Vec Skip-Gram Implementation

  **Overview**
This repository contains a localized, PyTorch-based implementation of the Continuous Skip-gram architecture as detailed in Mikolov et al.'s paper, Efficient Estimation of Word Representations in Vector Space. 

The objective of this task is to prove the mathematical soundness of a log-linear model (lacking a non-linear hidden layer) for learning continuous word vector representations.

 **Repository Structure****
PAPER_NOTES.md: Contains the academic analysis, central claims, architectural breakdowns, and the justification for the evaluation constraints.
src/word2vec_toy.py: The complete PyTorch implementation of the Continuous Skip-gram model.
results: Contains execution logs and screenshots verifying the model's successful training loop, loss reduction, and vector output.

 **Execution Constraints & Dataset**
The original paper trained a 300-dimensional vector space on a 6-Billion word corpus using distributed data centers. Because replicating this scale (or even a standard subset like text8) locally results in severe hardware bottlenecks, this implementation utilizes a heavily constrained toy dataset ("the quick brown fox..."). 

This intentionally limits the model's ability to learn complex semantic analogies (like the famous `King - Man + Woman = Queen`), but successfully isolates and proves the core mechanics: the projection layer, the Skip-gram context targeting, and the Cross-Entropy loss optimization.

## How to Run
### Dependencies
This script requires Python 3 and PyTorch. 
`pip install torch`

### Execution
Navigate to the `src` directory and run the python script:
`cd src`
`python word2vec_toy.py`

### Expected Output
1. A progression of epochs demonstrating the reduction of the Cross-Entropy Loss.
2. The raw 50-dimensional tensor vector extracted for a target word.
3. The model's highest-probability context guess based on the localized training data.
