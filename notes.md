# Natural Language Processing
Super essential notes on the whole syllabus.

## 1. Vector Space Models

The goal of NLP is to have machines understand text. To do so, we clearly need to turn it into numbers.

The process works in two steps:
1) Split the text into units, called "tokens".
2) Transform tokens into numbers.

### Tokenization

Tokenizing means breaking a sentence into atomic units.
The definition of unit we choose to adopt conditions the size of the vector space in which we can represent documents, as well as the density of said space.
We shall aim for a balanced vector space.

There are 3 main types of tokenizers:
1) Heuristic-Based Tokenizers, based on rules that depend on the language we are considering. For example, we may choose to cut all prefixes and suffices (a process called "Stemming").
This approach is of course very tedious, as we need to define all rules accordingly.

2) Linguistic-Based Tokenizers, which use a large vocabulary built using rules and capable of identifying lemmas ("be" -> "is"), parts of speech, ...
This approach is less tedious, as we use an ontology built by someone else. However, it is not specific to the text we are tokenizing and there is no way of tuning vocabulary size.

3) Learning-Based Tokenizers, which run an algorithm on a large piece of text and, starting from single letters, it combines pairs of frequently consecutive tokens. This is what WordPiece does.

### Tokens into Numbers

Transforming tokens into numbers requires choosing a Model of the data. The idea is that we can:
- Tokenize a document.
- Represent it using the model of choice.
- Do any other type of task (classification/regression/generation/...) using the numeric representation.

A simple model is a Bag of Words (BoW). It maps each token with a single number.
There are many ways to choose that number:
- It could simply be the number of times it appears in the document (maybe normalized).
In general, we would like tokens to have a high value if:
- They appear often in the text.
- Their frequent appearence is not expected, therefore it is meaningful.
To do so, we use TF-IDF:
- TF stands for Term Frequency. It's how often the word appears in the document.
- IDF stands for Inverse Document Frequency. It's how often the word appears in any document.
The idea is simple: words that are both locally frequent and globally rare are the most meaningful.

Common limitations of a BoW with Tf-Idf are:
a) Sparsity, as vectors are mostly zeros (most of the tokens are likely not going to appear in a document - unless it's really large and vague).
b) No ordering, as the model does not represent it in any way.
c) No semantics, as any pair of tokens is equally different.


## 2. N-Grams & Markov Language Models

The issue we want to tackle first is the ordering. To do so, we view the problem of analyzing text under a different perspective: a token-by-token generation.
The idea is that we want to determine the probability that a certain token appears after a sequence of other tokens.

The idea of defining a probability distribution over the next token in a sequence is what we call a Language Model.

We could think of representing this probability as absolute:
$
P("How\ are\ you") = P("How") \cdot P("are") \cdot P("you")
$
This is clearly not a good interpretation, as once again it does not account for tokens ordering.

A better idea is to use conditional probability:
$
P("How\ are\ you") = P("How|START") \cdot P("are|How, START") \cdot P("you|are, How, START")
$
This finally accounts for ordering, but it is clearly unfeasible for large sequences. In fact, if we want to compute $P("you|are, How, START")$, we should count all instances of "START How are you" in the document we are considering, which are clearly going to be not that many.

We make the assumption of tokens only depending on the k previous tokens in the sequence. This is a Markov Assumption.
For example, k=1:
$
P("How\ are\ you") = P("How|START") \cdot P("are|How") \cdot P("you|are")
$
Now we simply have to count the occurences of each pair and divide by the occurences of each token, singularly.

Token generation is a Maximum Likelihood Estimation.

Markov Language Models have some issues:
a) Sparsity (due to zero-probability), as we may want to compute P("A|B") without ever having seen "A B" in our reference document.
b) Short Memory, as we clearly have no way to express probabilities for long-term dependencies. If we say "I have not seen Alice in a while, I wonder how ..." the model will not account for "Alice" and will have equal likelihood of generating "she" and "he" as next tokens.
c) No semantics, as once again each pair of tokens is equally different.

## 3. Bengio's NPLM and Word2Vec

After many years of research, Bengio et al. finally tackle the issue of semantics by building the first Neural Probabilistic Language Model.
The idea is to use a Neural Network:
- The Input is a One-Hot Encoded token. The size of this vector is the size of the dictionary.
- The Output is a probability distribution over the next token.
- The network would:
    - In the 1st layer, transform the OHE token into a vector representation. This is done simply with a matrix multiplication (that extracts the vector representation from a matrix). The lines within the matrix are called Embeddings.
    - In the 2nd layer, the network transforms the embedding into the internal representation for the next token generation (it's just logits).
    - Finally, the logits are transformed into probabilities, thus building the probability distribution for the appearence of the next token.
This formulation is equivalent to a Markov Chain with k=1. The original network could be extended to any k.

Bengio's network could be trained using standard gradient descent and we could have a working Neural Language Model.
However, this was not as big of a step as it could have been: as the vocabulary gets bigger, the number of the parameters in the network explodes. In short, the network is extremely slow and requires very large amounts of data to learn anything.

10 years later, Mikolov et al. finally find a solution to the efficiency problem, by simplifying Bengio's network and dropping the output layer: after all, the real power was in the embeddings it produced.
The model, called Word2Vec, would be trained using Skip-grams and CBOW as "proxy tasks":
- Let's consider the text "Clouds are in the sky, this is not a sunny day". The skip-gram for the token "sky" is just the tokens that surround it (both before and after) - it's a representation of its context.
- 