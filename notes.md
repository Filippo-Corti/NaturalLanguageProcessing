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

10 years later, Mikolov et al. finally find a solution to the efficiency problem, by simplifying Bengio's network and dropping the idea of generating tokens in a sequence: after all, the real power was in the embeddings it produced.
The model, called Word2Vec, can be trained using Skip-grams and CBOW as "proxy tasks" (basically, same network architecture but two different objective functions). Let's consider "The cat sits on the table":
- The CBOW task consists in predicting "sits" given its context "the, cat, on, the".
- The Skip-grams task consists in prediction the context "the, cat, on, the" given the center word "sits".
These proxy-tasks are just means to make sure that the produced embeddings are solid representations of the meaning of words (under the assumption that meaning is similarity with other words).

By using Word2Vec to build embeddings, we managed to represent the meaning of words. Interestingly, we can compute operations between vectors in the embeddings vector space and they will make sense to us (things like 'king' - 'man' + 'woman' = 'queen').

The limitation that we could not solve just by using embedding was:
a) Short Memory, as once again we were limiting our understanding of the context to a fixed window, just like we did with n-grams.

## 4. Recurrent Neural Networks (RNNs)

The main idea behind the introduction of a long term memory in neural networks is to exploit recursion: after the network has generated a new token, it should preserve its hidden state (h) when generating the next one. The hidden state represents the memory of the network.
In its simplest form, a RNN has, as weights:
- Those that transform the new input into a new hidden state.
- Those that transform the old hidden state into the new one.
- Those that transform the hidden state into the output.
The main issue with RNNs was gradient vanishing and explosion, particularly present due to BPTT, required for training. Basically, as we extend the flattened network, long term dependencies become weaker and weaker.

More refined versions of RNNs try to fix this issue.
### Long-Short Term Memory (LSTMs)

LSTMs represent their memory through two distinct vectors:
- The long term memory c.
- The short term memory h.
Both memories are handle using gates. In particular:
- The forget gate controls how the long term memory c should be erased, looking at the new information. Its output is a filter to be applied to c.
- The input gate controls how the long term memory c should be updated, looking at the new information. Its output is a filter to be applied to g = tanh(x + h) - which represents the actual new information coming from the new token x.
- The output gate controls how the output y and the new short term memory h should be produced starting from the current memories.

LSTMs try to avoid vanishing gradients because they introduce addition to memory.

### Gated Recurrent Unit (GRUs)

GRUs try to embed in the same vector both long-term and short-term memory, in an attempt to simplify LSTMs.
They use:
- The update gate, which combines forget and input gate into a single one.
- The reset gate, which controls how much of the past information is used to compute the new candidate.

They are slightly faster than LSMTs. 
Overall, however, both models have three big issues:
a) Very Long Term Dependencies are still hard to model, due to vanishing gradients.
b) They require Sequentiality, therefore they cannot be parallelized at all. This makes them GPU unfriendly and thus slow.
c) Inputs and Outputs are always composed by the same number of tokens. This is because each output token comes from one input token. This is not very effective for many sequence-to-sequence tasks (such as translation).

## 5. Encoder-Decoder Architectures

In order to allow input and output to have different length, we need to separate:
- The Encoder, which is the part of the network that reads the input sequence in order and builds up the hidden state "h", which then becomes the context "c" of the input.
- The Decoder, which receives the context "c" and is responsible for generating the output sequence, one token at a time.
Both architectures are singular RNNs. Thanks to the separation, the output length is not necessarily the input length anymore.

A quick note on how the decoder transforms the probability distribution over the tokens into the output sequence:
- Ideally, our goal is to determine the **entire sequence** that maximizes the probability of appearing after the full input sequence of tokens. This means we should evaluate probability for all possible V^n sequences of tokens. As this is unfeasible, we resort to other techniques.
- Greedy Search assumes that probabilities of each token in the output sequence is independent, so we pick the best choice locally. Note that the best token at step T does not necessarily correspond to the token appearing in spot T of the full sequence maximizing the probability.
- Beam Search expands a tree of options by picking, at each step T, the n options with highest probability. AT the end, we can pick the branch of the tree with the highest joint probability.

A known method to evaluate sequence-to-sequence generation for the task of translation (but also in general) is BLEU (Bilingual Evaluation Understudy), which compares n-grams between the input and output sequences.

### Attention

An issue with the described Encoder-Decoder Architecture is conceptual: we are asking the Encoder to represent a full sentence in a single vector, and we are asking the Decoder to extract a full sentence from just one vector. It is clearly not a scalable model.

To address this issue, Bahdanau et al. have introduced the mechanism of Attention: we provide the decoder with a mechanism that makes it possible to use all hidden representations produced by the encoder at each time step (and not just the last one, as we did before).
The Attention mechanism requires to compute a relevance score between the current internal state of the decoder and all the internal states of the encoder. Then, the scores are turned into weights using a softmax, which is then applied to the encoder states to discount their relevance.

Notice that attention is fully learnable by the network, as it is composed of weight matrices.

Even with attention, we still had some issues to solve:
a) Sequentiality was still required, as both encoder and decoder were RNNs.
b) Long memory was possible, but quite costly.

## 6. Transformers

Vaswani et al. finally solved the computational struggle of sequentiality by discarding completely the idea of RNNs and letting the network learn based solely on attention. We could employ attention in two forms:
- Self-Attention, which could be used to compute attention scores across the same set of tokens.
- Cross-Attention, which could be used to compute attention scores between the encoder's and the decoder's vectors.

This made it possible to fully parallelize the operations of the network, which would consider all tokens of the input at once.
At the same time, this meant that we were losing the possibility to represent ordering. This was added back using a Positional Encoding operation, which added to the input embeddings some values that could distinguish the same token appearing multiple times in the input sequence, based on the position it appeared in.

Some important details are:
- Multi-Head Attention, to allow the model to jointly attend to information from different representation subspaces. In other words, each head learns different aspects of the relationship betwen the inputs.
- Stacked Encoder and Decoder: as we removed any sequentiality, we could actually add it back by running through multiple encoders and decoders, in a sequence.
- Masked Attention, to avoid (during training) the decoder attending to tokens that it should not have seen yet.
- Residual Connections, to avoid gradient vanishing and avoid that the attention deconstructs the inputs eccessively.

The only limitation of transformers was the complexity of the computation of the attentions, which was quadratic. This made long documents expensive.

## 7. BERT & GPT

In the modern era, the encoder and decoder from the transformers architecture have been separated into two different architectures, specialized on one of the two tasks:
- BERT, Encoder-only.
- GPT, Decoder-only.
The driving idea for this change is that both encoder and decoder have a sort of language understanding, so we likely do not need both.

### Bidirectional Encoder Representations from Transformers (BERT)

BERT only uses the Encoder stack from the Transformers architecture. Its goal is not to generate text, but to understand it.
Its architecture is composed of:
- Embeddings
- Multi-Head Self-Attention
- FFN
- Residual Connections and Normalizations

A great property of BERT is transfer learning: we can pre-train it to understand language in general, and then fine-tune it to solve specific tasks at low computational cost.

BERT can be trained on two tasks:
1) Masked Language Modeling (MLM), where we replace some tokens in the input with the token \[MASK\] and train BERT to predict only the masked tokens.
There are two issues with this naive strategy:
- Mismatch between pretraining and fine-tuning, as the model never sees \[MASK\] for downstream tasks.
- Shortcut learning, as the model might only focus on the MASK and ignore all surrouding context, as it knows it is always right.
We therefore employ an 80-10-10 procedure:
- We select 15% of all tokens. Among those:
    - 80% become \[MASK\]
    - 10% become a randoom token.
    - 10% stays as it is.
The idea is to introduce noise so that the model actually has to learn the contextual representation for all tokens, not just how to fill the MASK blocks.

2) Next Sentence Prediction (NSP), where we provide the model with two sentences A and B and ask it to predict IS_NEXT or NOT_NEXt, corresponding to whether B is the logical continuation for A.
It is great for Q&A and NLI.

BERT uses by default and embedding size of 768, 12 encoder layers and a total of 108M params.


### Generative Pretrained Transformer (GPT)

GPT only uses the Decoder stack of the transformer architecture. Its purpose is purely to generate text: that is, predict the next token given all previous tokens.

With respect to the original decoder architecture:
- Masked Self-Attention is maintained.
- There is no Cross-Attention.
- Everything else stays the same.

GPT is trained on the next token prediction task, with standard Cross Entropy Loss.


## Other topics

### Knowledge Distillation

Formalized first by Hinton, it is inspired by enseble neural network medhos, which uses multiple modes as training and then compressing into a single one.
Knowledge Distillation is a model compression and transfer technique that allows a smaller model (student) to leran from a larger model (teacher). It aims to transfer the knowledge encoded in the teacher into a more efficient one, without significantly losing predictive performance.

- Logits / Soft-Target Distillation
- Response-Based Distillation
- Feature Distillation
...


### Explainability

By XAI we mean the need of a description of the reasons behind the machine's behaviour that is understandable in human terms.

Explainability may be:
- Local, aiming to clarify why a model made a specific decision for a particular input.
- Global, aiming to descirbe the overall behavior of the model across all its inputs.
Local explainations provide details, global explainations provide general rules and patterns.

#### SHAP Methods

SHAP assigns each feature a contribution value for a particular prediction. 
It is based on game theory, where each feature is seen as a player in cooperative game and model's prediction is the payout.

SHAP calculates how the prediction changes when a feature is added or removed, averaging over many combinations.
That is, it measures how much each feature changes the prediction on average, across all possible subsets of other features.
It guarantees:
- Additivity: the SHAP values across all features sums to the total difference between actual and expected.
- Consistency: if a model changes to increase feature's contribution, its SHAP value will not decrease.
We can think of the features to be the tokens of the input sequence: we measure how much it would impact changing that token to another one.

#### Saliency Maps

Saliency Maps are a simple and intuitive method to visualize which parts of an input influences a neural network's prediction the most.
They highlight regions of the input where the model is most sensitive to changes.

Mathematically, it's just a partial derivative with respect to the input.
SHAP is perturbation based, Saliency is based on gradient.

#### Concept-Activation Vectors

Concept Activation Vector = weight vector from a logistic regression separating concept vs non-concept examples in a specific layer’s activation space.

### Biases

A Bias is a tendency or preference that causes unfair or unbalanced outcomes. In AI and machine learning, it happens when a model makes decisions that systematically favor or disadvantage certain groups, ideas or outcomes due to the data it was trained on.
Biases can enforce stereotypes.


