**1) CENTRAL CLAIM**
The paper asserts that simple log-linear models can learn high-quality continuous word vector representations from massive datasets (billions of words) significantly faster and at a much lower  cost than complex neural network language models . Furthermore, these efficiently trained vectors achieve state-of-the-art performance in capturing both syntactic and complex semantic regularities (e.g., answering "vector(King) - vector(Man) + vector(Woman) = vector(Queen)").

**2)Core Architecture / Algorithm**
To test this claim, one of two novel log-linear architectures must be implemented:

a) **Continuous Bag-of-Words (CBOW):** A model that predicts the current (middle) word based on its surrounding context (e.g., 4 future and 4 history words). The context words are projected into the same position (averaged) in a shared projection layer without a non-linear hidden layer.

b) **Continuous Skip-gram:** A model that takes the current word as input and tries to predict the surrounding context words within a certain window (e.g., a range of 10 words). Distant words are sampled less frequently during training to give them less weight.

**Implementation Details:** Both models rely heavily on a hierarchical softmax normalization (using a Huffman binary tree based on word frequencies) to reduce the computational bottleneck of the output layer.

**3) Dataset, Evaluation Metric, and Baselines**
**Dataset:** The primary training dataset is the Google News corpus (containing roughly 6 billion tokens), with the vocabulary restricted to the 1 million most frequent words.

**Evaluation Metric:** The models are evaluated on a custom Semantic-Syntactic Word Relationship test set (8,869 semantic and 10,675 syntactic questions). Accuracy is measured using a vector offset calculation (e.g., X = vector("biggest") - vector("big") + vector("small")) and finding the closest exact word match via cosine distance.




