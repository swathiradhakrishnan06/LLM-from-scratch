# Large Language Models (LLM) From Scratch

> _This content is inspired by Dr. Raj Dandekar’s YouTube playlist on Large Language Models. It will be updated as and when more topics are covered._

---

## 📑 Table of Contents

1. [📘 Large Language Models (LLM) Basics](#-large-language-models-llm-basics)
2. [🧪 Pretraining LLMs vs Finetuning LLMs](#-pretraining-llms-vs-finetuning-llms)
3. [🤖 Understanding Transformers and LLMs](#-understanding-transformers-and-llms)
4. [📚 The Evolution and Core Mechanics of GPT Models](#-the-evolution-and-core-mechanics-of-gpt-models)
5. [🧱 Stages of Building a Large Language Model (LLM) from Scratch](#-stages-of-building-a-large-language-model-llm-from-scratch)
6. [🧩 Code an LLM Tokenizer from Scratch in Python](#-code-an-llm-tokenizer-from-scratch-in-python)
7. [🧠 GPT Tokenisation using Byte Pair Encoding (BPE)](#-gpt-tokenisation-using-byte-pair-encoding-bpe)

---

## 📘 Large Language Models (LLM) Basics

### 1. What Exactly is a Large Language Model (LLM)?

At its simplest, **an LLM is a neural network designed to understand, generate, and respond to human-like text**. Key components include:

* **Neural Network**:
  Deep neural networks, inspired by the human brain, consist of layers of interconnected "neurons" that process input data. These networks are used for tasks such as:

  * Image detection
  * Text generation
  * Autonomous vehicles

* **Human-like Text Processing**:

  * **Understanding**: Comprehending human text.
  * **Generating**: Creating new text.
  * **Responding**: Interacting based on input prompts.

> *Example*: ChatGPT can understand prompts like “plan a relaxing day” and respond conversationally with a detailed itinerary.

> *In short*: LLMs are **deep neural networks trained on massive datasets** to perform tasks like understanding, generating, and responding to human-like text — often in a human-like way.

---

### 2. The Meaning of “Large” in LLM

The "large" refers to the **number of parameters** in the model.

* **Scale**:

  * Models like GPT-3 have up to **175 billion parameters**.
  * GPT-4 has **even more**.
* **Growth**:

  * There has been an **exponential increase** in model size, especially post-2020.
  * Language models dominate this growth.

> The sheer **scale of parameters** differentiates LLMs from earlier, smaller models.

---

### 3. The Meaning of “Language Model”

A **language model** only deals with **textual data**, not images or videos.

LLMs are capable of performing various **Natural Language Processing (NLP)** tasks such as:

* Question Answering
* Translation
* Sentiment Analysis
* Text Completion

---

### 4. Difference Between Modern LLMs and Earlier NLP Models

#### 🆚 Earlier NLP Models:

* **Task-Specific**: One model = one task (e.g., only translation or only sentiment analysis)
* **Limited Capabilities**: Could not handle complex instructions like generating emails.

#### ✅ Modern LLMs:

* **General-Purpose**: One model = many NLP tasks.
* **Highly Capable**: Can write emails, generate articles, translate, etc.
* **Extremely Flexible**: Applications are broad and still expanding.

---

### 5. The "Secret Sauce": Transformer Architecture

The **Transformer** is the breakthrough architecture behind LLMs.

* **Origin**: Introduced in the 2017 paper **“Attention Is All You Need”** by Google Brain.
* **Impact**: Over **100,000 citations** — revolutionized AI.
* **Core Concepts**:

  * Input Embeddings
  * Multi-Head Attention
  * Positional Encoding

> While complex, understanding the Transformer is **essential to grasping how LLMs work**.

---

### 6. Understanding AI, ML, DL, LLM, and Generative AI Terminologies

#### 🤖 Artificial Intelligence (AI)

> Any system that mimics human intelligence.
> Includes rule-based systems (e.g., scripted chatbots).

#### 📊 Machine Learning (ML)

> Subset of AI.
> Learns and adapts based on data and user interactions. Includes neural networks, decision trees, etc.

#### 🧠 Deep Learning (DL)

> Subset of ML using **neural networks only**.
> Examples: image classifiers, speech recognizers.

#### 📝 Large Language Models (LLMs)

> Subset of DL for **text-based applications only**.
> Do **not** handle images or videos.

#### 🧬 Generative AI

> Combines DL + LLMs to **generate new content**.
> Covers multiple media types: text, images, videos, sound, etc.

#### 📌 Summary Hierarchy:

```
AI > ML > DL > LLM
Generative AI = DL + LLM + multimodal content generation
```

---

### 7. Applications of LLMs

LLMs are widely used across many domains. Key application areas:

#### ✍️ Content Creation

* Poems, books, news articles, social media content.

#### 🤖 Chatbots & Virtual Assistants

* Used in banks, airlines, restaurants for automated customer support.

#### 🌐 Machine Translation

* Translates across languages, including regional ones.

#### 🆕 New Text Generation

* Generates original text content on any topic.

#### 😊 Sentiment Analysis

* Detects tone, mood, or hate speech from text.

#### 🛠️ Additional Tools Enabled by LLMs:

* YouTube Script Generator
* MCQ Generator
* Text Summarizer
* Text Rewriter
* Lesson Plan Generator

> These tools save time and effort in education, media, business, and more.

---

## 🧪 Pretraining LLMs vs Finetuning LLMs

### 1. Overview of LLM Building Stages

Building an LLM primarily involves **two distinct but interconnected stages**:

* **Pre-training**: Teaches the model to understand and generate human-like language using a broad and diverse dataset.
* **Fine-tuning**: Adapts the model to specific tasks, domains, or industries using narrower, task-specific datasets.

---

### 2. Pre-training: The Foundational Stage

#### 2.1. Definition and Purpose

Pre-training is the **initial and most resource-intensive phase** where the model learns the structure of human language. Initially, LLMs were trained for **word completion**—predicting the next word in a sentence.

> "How is it able to interact so effectively with humans? How can it understand and respond so accurately?"

#### 2.2. Data Requirements

Pre-training requires **unlabeled, raw text data** from a variety of large-scale sources:

| Dataset      | Description                         | Size              |
| ------------ | ----------------------------------- | ----------------- |
| Common Crawl | Open internet data                  | 410 billion words |
| WebText2     | Reddit, blogs, Stack Overflow, code | 20 billion words  |
| Books        | Digitized book content              | 67 billion words  |
| Wikipedia    | Encyclopedia articles               | 3 billion words   |

> Example: GPT-3 was trained on **300 billion tokens** (≈ 300 billion words).

This is similar to how **children learn from parents and surroundings**, absorbing knowledge from all interactions.

#### 2.3. Computational Cost and Scale

* Extremely resource-heavy: **GPT-3’s pre-training cost was \~\$4.6 million**
* Requires **powerful GPUs or TPUs**
* Not feasible for individuals or small teams

> “Not possible for normal students or even for people without access to enterprise infrastructure.”

#### 2.4. Capabilities of Pre-trained Models

Pre-trained models (a.k.a. **foundational models**) generalize well across many tasks **without being explicitly trained for each**:

* Text translation
* Question answering
* Sentiment analysis
* Summarization
* Linguistic acceptability
* Generating MCQs

> "One model can do all of these tasks on its own without ever being trained for these tasks."

---

### 3. Fine-tuning: Specialisation and Refinement

#### 3.1. Definition and Purpose

Fine-tuning involves **adapting a pre-trained model** to perform a **specific task or serve a specific industry**.

> "It’s a refinement on pre-training using a much narrower, labeled dataset."

#### 3.2. Data Requirements

Fine-tuning uses **labeled datasets**, such as:

* **Instruction fine-tuning**: e.g., pairs like (Prompt, Ideal Response)
* **Classification**: e.g., emails labeled as spam vs non-spam

#### 3.3. Use Cases and Examples

| Company            | Application              | Outcome / Notes                                          |
| ------------------ | ------------------------ | -------------------------------------------------------- |
| ✈️ Airline Company | Customer Support Bot     | Responses specific to their own flight data & policies   |
| 📘 Educational Co. | Question Generation Tool | High-quality, subject-specific questions                 |
| ☎️ SK Telecom      | Korean Telecom Bot       | +35% conversation summarization, +33% intent recognition |
| ⚖️ Harvey (Legal)  | Legal Assistant          | Trained on case law & legal documents                    |
| 🏦 JP Morgan Chase | Enterprise AI Assistant  | Uses internal data; not possible with GPT alone          |

> "Foundational models are good for general users; industries **require** fine-tuning."

#### 3.4. When is Fine-tuning Necessary?

* When moving to **production-level deployment**
* For **startups or enterprises** with proprietary data
* For **specialised accuracy**, personalization, or regulatory needs

> "Big companies never use just the foundational model—they always take the next step of fine-tuning."

---

### 4. Schematic Overview of LLM Building

```
🧩 DATA COLLECTION
    ⬇
🛠️ PRE-TRAINING (Foundational Model)
    - Unlabeled data
    - Expensive, resource-intensive
    - General-purpose capabilities
    ⬇
🎯 FINE-TUNING (Custom Model)
    - Labeled, narrow domain data
    - Task-specific or company-specific outputs
```

> Result: A **fine-tuned LLM** capable of real-world applications like summarizers, translators, legal bots, or customer service assistants.

---

## 🤖 Understanding Transformers and LLMs

### 1. The Transformer: The “Secret Sauce” of Modern LLMs

* Transformers are the **core architecture** powering most modern Large Language Models (LLMs).
* Introduced in the landmark 2017 paper *["Attention Is All You Need"](https://arxiv.org/abs/1706.03762)*, which has received **100,000+ citations**.
* Originally developed for **machine translation tasks** (e.g., English to German), not for text generation.
* GPT (Generative Pre-trained Transformer)—the foundation of ChatGPT—is **built on the Transformer architecture**.

---

### 2. Simplified Transformer Architecture: 8-Step Flow

An intuitive walkthrough of how a Transformer translates text, e.g., English to German:

1. **Input Text**

   * Example: “this is an example”

2. **Pre-processing (Tokenization)**

   * The input is split into **tokens** and mapped to **IDs**
   * Note: Tokens are not always equivalent to words.

3. **Encoder & Vector Embedding**

   * Each token is converted into a **high-dimensional vector**, capturing **semantic relationships**
   * Example: Vectors for *“dog”* and *“puppy”* are close together

4. **Generated Vector Embeddings**

   * Output from encoder: contextualized vector representations of input

5. **Partial Output Text (Decoder Input)**

   * Decoder receives the partially generated output (e.g., “Das ist…”)

6. **Decoder Input**

   * Decoder takes both **encoder embeddings** and **partial output** to continue generating

7. **Word Prediction by Decoder**

   * Decoder predicts the **next word/token** based on context

8. **Final Output**

   * Full translated sentence is generated
   * Learning is improved via a **loss function**

---

### 3. Key Components: Encoder, Decoder & Self-Attention

| Component          | Purpose                                                          |
| ------------------ | ---------------------------------------------------------------- |
| **Encoder**        | Converts input tokens into contextual embeddings                 |
| **Decoder**        | Generates output based on embeddings and prior output            |
| **Self-Attention** | Core innovation: enables model to focus on relevant words/tokens |

#### 🔍 Self-Attention Explained:

* **Assigns attention scores** to input tokens based on importance for predicting the next word
* Enables understanding of **long-range dependencies**

  > Like remembering what happened **pages ago** in a novel
* Helps model decide **which words to "attend to"** at each step

---

### 4. Transformer Variants: BERT vs GPT

#### 🧠 BERT (Bidirectional Encoder Representations from Transformers)

* **Task**: Predict masked words in a sentence (Masked Language Modeling)
* **Bidirectional**: Looks at context **before and after** each word
* **Architecture**: Uses **only the encoder**
* **Use Cases**: Sentiment analysis, classification tasks

#### ✍️ GPT (Generative Pre-trained Transformer)

* **Task**: Predict the **next word** in a sequence (causal language modeling)
* **Unidirectional**: Left-to-right only
* **Architecture**: Uses **only the decoder**
* **Use Cases**: Text generation, summarization, chatbots (e.g., ChatGPT)

| Feature        | BERT                | GPT                            |
| -------------- | ------------------- | ------------------------------ |
| Directionality | Bidirectional       | Unidirectional (left to right) |
| Component Used | Encoder only        | Decoder only                   |
| Use Case       | Understanding tasks | Generative tasks               |

---

### 5. Transformers ≠ LLMs

> ⚠️ The terms “Transformers” and “LLMs” are often **confused** but are **not the same**.

#### Not All Transformers Are LLMs:

* Transformers can be used in **non-language** domains like vision:

  * Example: **Vision Transformers (ViT)** outperform CNNs in tasks like:

    * Image classification
    * Tumor detection
    * Road pothole detection

#### Not All LLMs Are Transformers:

* LLMs can be built using **other architectures** like:

  * **RNNs (Recurrent Neural Networks)** – introduced in 1980
  * **LSTMs (Long Short-Term Memory)** – introduced in 1997
* These models handled **sequence prediction** tasks and had memory mechanisms:

  * **RNNs**: Feedback loop for short-term memory
  * **LSTMs**: Paths for both short-term and long-term memory

#### ✅ Summary:

* **Transformer** = An architecture (neural network design)
* **LLM** = A model that understands/generates language (can use different architectures)

> **Most modern LLMs use Transformer architecture, but not all Transformers are LLMs.**

---

## 📚 The Evolution and Core Mechanics of GPT Models

### 1. 🕰️ Historical Progression of GPT Models

#### 🔹 Transformers (2017): “Attention Is All You Need”

* Introduced **self-attention mechanism** to model long-range dependencies.
* Replaced RNNs and LSTMs in many NLP tasks.
* Architecture had both **encoder and decoder** blocks.

#### 🔹 GPT-1 (2018): *“Improving Language Understanding with Unsupervised Learning”*

* Introduced the **Generative Pre-training** paradigm.
* Removed the encoder – GPT models are **decoder-only**.
* Trained on large-scale **unlabeled text** to predict the **next word** (unsupervised learning).

#### 🔹 GPT-2 (2019): *“Language Models are Unsupervised Multitask Learners”*

* Scaled up both **model size** and **dataset**.
* Largest version had **1.5 billion parameters**.
* Demonstrated **multi-task** capabilities without task-specific training.

#### 🔹 GPT-3 (2020): *“Language Models are Few-Shot Learners”*

* Massive scale-up to **175 billion parameters**.
* Showed **few-shot**, **one-shot**, and **zero-shot** learning capabilities.
* Could translate, answer questions, recognize emotion, etc., with **few or no examples**.

#### 🔹 GPT-3.5 and GPT-4 (2022–Present)

* GPT-3.5: Sparked **commercial virality** (ChatGPT release).
* GPT-4: Represents the **state-of-the-art** with improved reasoning and accuracy.

---

### 2. 🧠 Zero-Shot, One-Shot, and Few-Shot Learning

| Learning Type | Description                                | Example                                                   |
| ------------- | ------------------------------------------ | --------------------------------------------------------- |
| **Zero-Shot** | Performs a task with **no prior examples** | Translate “cheese” → “fromage” with only task instruction |
| **One-Shot**  | One example provided                       | “Sea otter → loutre de mer” before translating “cheese”   |
| **Few-Shot**  | Few examples provided                      | 3–5 translation examples before asking model to translate |

> 📝 GPT-3 and GPT-4: Primarily **few-shot learners**, but also excel in **zero-shot** settings.

---

### 3. 📊 Data Scale and Training Costs

#### 📁 Dataset Sources

* **300 billion tokens** used for GPT-3 training.
* Major components:

  * **Common Crawl**: 410B tokens (\~60%)
  * **WebText2**: 19B tokens (\~22%)
  * **Books + Wikipedia**: \~18%

#### 🔢 Tokenization

* A **token** is the model’s smallest meaningful unit – not always equal to one word.

#### 💰 Training Cost

* **\$4.6 million** estimated cost for GPT-3 pre-training.
* Driven by:

  * Size: **175B parameters**
  * Resource: **GPU/TPU compute**

#### 🧪 Pre-training vs. Fine-tuning

| Stage            | Description                                                                         |
| ---------------- | ----------------------------------------------------------------------------------- |
| **Pre-training** | Unsupervised learning on **diverse, large datasets**                                |
| **Fine-tuning**  | Supervised learning on **narrow, domain-specific data** for real-world applications |

---

### 4. 🏗️ GPT Architecture and Auto-Regressive Nature

#### 🔧 Architecture

* GPT uses **only the decoder** from the Transformer.
* GPT-3 includes **96 Transformer layers**.

#### 🔁 Auto-Regressive Property

* Predicts text **one token at a time** using previous tokens.
* Previous output becomes part of the **input** for the next prediction.

#### 🧩 Self-Supervised Learning

* Label = next word in sentence → No need for manual annotations.
* Trained by minimizing difference between **predicted** and **true** next word.

---

### 5. 🌟 Emergent Behaviour

#### 📌 Definition

> "The ability of a model to perform tasks it wasn’t explicitly trained for."

#### 💡 Surprising Capabilities

* Language translation
* Essay grading
* Multiple-choice question generation
* Text summarization
* Emotional recognition

#### 🔍 Research Frontier

* Emergent abilities remain **not fully understood**.
* A major area of **active research** in AI.

---

### 6. 🧬 Open-Source vs. Closed-Source Models

| Model Type        | Description                                             |
| ----------------- | ------------------------------------------------------- |
| **Closed-Source** | Parameters are hidden; access via API (e.g., GPT-4)     |
| **Open-Source**   | Full model weights publicly available (e.g., LLaMA 3.1) |

#### 📈 Performance Shift

* LLaMA 3.1 has **405B parameters** and **approaches or beats GPT-4** in benchmarks.
* **Open-source is catching up** in performance and accessibility.

---

### 🧠 Summary

* GPT models evolved from **decoder-only** variants of Transformers, optimized for **next-word prediction**.
* Their scale enables **few-shot learning** and **emergent behaviour** across tasks.
* While GPTs are currently **closed-source** in OpenAI’s offerings, **open-source alternatives** are rapidly closing the gap.
* GPT’s power lies in its **auto-regressive, unsupervised training** and **self-attention mechanism**, which together allow surprisingly **versatile capabilities** without task-specific training.

---

## 🧱 Stages of Building a Large Language Model (LLM) from Scratch

### 1. 🚧 Three-Stage Approach to LLM Development

The development of an LLM follows a **structured 3-stage pipeline**, from raw data to production-level applications.

---

#### 🔹 **Stage 1: Building Blocks**

> Focus: Preparing data, understanding architecture, and implementing attention mechanisms.

##### 📌 Data Preparation & Sampling

* **Tokenisation**: Split sentences into tokens (smallest units of meaning).
* **Vector Embedding**: Convert words into vectors that reflect semantic meaning.

  * E.g., “apple,” “banana,” and “orange” lie close in vector space.
* **Positional Encoding**: Adds word order info, essential for capturing sentence structure.
* **Batching Data**: Organizes data into manageable units for efficient training on the **next-word prediction** task.

##### 🤖 Attention Mechanism

* Core component of Transformer models.
* Allows the model to focus on **relevant parts of the input**, not just immediate neighbors.
* Supports understanding **context across the entire input sequence**.
* Includes:

  * **Multi-head attention**
  * **Masked multi-head attention**
  * **Input/output embeddings**

##### 🏗️ LLM Architecture

* Involves stacking multiple Transformer layers and attention heads.
* GPT uses a **decoder-only architecture** (unlike original Transformer which had both encoder and decoder).

---

#### 🔹 **Stage 2: Pre-training – Building the Foundational Model**

> Focus: Training on large-scale **unlabeled data**.

##### ⚙️ Pre-training Process

* Define **epochs**, compute **loss gradients**, and update **model parameters**.
* Generate sample outputs at intervals to evaluate performance.

##### 💾 Efficiency Tip

* **Save and load weights** after pre-training to avoid full retraining.
* Example: Load pre-trained weights from **OpenAI** into your model.

##### 🎯 Objective

* Train a **general-purpose foundational model** that understands language broadly.

---

#### 🔹 **Stage 3: Fine-tuning – Building Specific Applications**

> Focus: Customizing the foundational model using **labeled data** for real-world tasks.

##### 🔑 Key Points

* Uses **task-specific labeled datasets** (e.g., spam vs. non-spam).
* **Fine-tuned LLMs** perform significantly better on specific applications than only pre-trained ones.

##### 🛠️ Example Applications

1. **Email classifier** – Detect spam.
2. **Chatbot** – Answer queries with contextual understanding.

##### 🧠 Why Fine-Tuning?

> "Nobody deploys the pre-trained model directly. You **fine-tune** it on your **specific labeled dataset** to make it production-ready."

---

### 2. 📚 Recap of Fundamental LLM Concepts

#### 🔁 Two-Step Training Process

| Step             | Description                                                    |
| ---------------- | -------------------------------------------------------------- |
| **Pre-training** | On large **unlabeled data**. Expensive, high-resource process. |
| **Fine-tuning**  | On **specific labeled data**. Focused and efficient.           |

#### 💥 Emergent Properties of LLMs

LLMs trained for next-word prediction develop surprising abilities like:

* Summarisation
* Translation
* Emotion recognition
* Question answering
* Multiple-choice generation

> These capabilities are **not explicitly programmed** — they *emerge* from scale and architecture.

---

#### 🧠 Transformers and Attention: The Secret Sauce

* The **attention mechanism** enables understanding of context and relationships between words.
* GPT models are built on the **Transformer architecture**, but only use the **decoder** portion.

| Model Type                      | Architecture      |
| ------------------------------- | ----------------- |
| **Original Transformer (2017)** | Encoder + Decoder |
| **GPT (2018 onwards)**          | Decoder-only      |

---

### 3. 🔑 Key Insights & Takeaways

* ✅ **Three-Stage Roadmap**: Data → Architecture → Pre-training → Fine-tuning.
* ✅ **Fine-tuning is Essential**: Real-world applications require adapting the model.
* ✅ **Emergent Abilities**: LLMs go far beyond basic next-word prediction.
* ✅ **Pre-training is Expensive**: E.g., GPT-3 cost **\$4.6 million** to pre-train.
* ✅ **Transformer + Attention = Power**: The architecture and attention mechanism underpin LLMs’ success.

---

Perfect! Here's the **final, complete markdown version** of your topic **"Understanding LLM Tokenisation"**, now including **code snippets inline with the corresponding explanation**, as per the Python code from your uploaded notebook.

---

## 🧩 Code an LLM Tokenizer from Scratch in Python

### 1. 🧠 Core Concepts of Tokenisation

Tokenisation is the **first and essential step** in building a Large Language Model (LLM). It transforms human-readable text into machine-readable numerical format.

> “Tokenisation is just the process of breaking down a sentence into individual words,”
> but in LLMs, it goes further — handling punctuation, spacing, and unknown tokens robustly.

---

### 2. 🔄 Tokenisation Pipeline

The tokenisation process can be broken into three steps:

1. **Splitting text** into word and subword tokens
2. **Converting tokens** into unique numerical token IDs
3. **Encoding token IDs** into vector representations (*not covered in this lecture*)

This section focuses on **steps 1 and 2**.

---

### 3. 🛠️ Building a Tokeniser from Scratch

It's recommended to implement a basic tokeniser manually for clarity.

#### Dataset:

* *Edith Wharton’s* **“The Verdict”** is used
* In real-world scenarios, LLMs are trained on **millions of documents**

---

#### ✂️ Step 1: Tokenising the Text

##### ➤ Split by whitespace and punctuation

```python
import re

def tokenize(text):
    tokens = re.split(r'([,.:;?_!"()\']|--|\s)', text)
    tokens = [item.strip() for item in tokens if item.strip()]
    return tokens
```

##### ✅ Handles:

* Word splits
* Punctuation as individual tokens
* Removes redundant whitespace

📌 Output on the full story: **4690 tokens**
Example output:

```python
['This', 'is', 'an', 'example', '.', 'Right', '?']
```

---

#### 🔢 Step 2: Creating a Vocabulary and Token IDs

Each unique token is assigned a unique integer (token ID).

```python
tokens = tokenize(text)
vocab = sorted(set(tokens))
vocab_dict = {token: idx for idx, token in enumerate(vocab)}
```

📌 Result: **Vocabulary size = 1130**

---

### 4. 🧰 The Tokeniser Class (V1)

This class encapsulates token-to-ID and ID-to-token mappings.

```python
class SimpleTokenizerV1:
    def __init__(self, vocab_dict):
        self.str_to_int = vocab_dict
        self.int_to_str = {i: s for s, i in vocab_dict.items()}

    def encode(self, text):
        tokens = tokenize(text)
        return [self.str_to_int[token] for token in tokens]

    def decode(self, token_ids):
        text = " ".join([self.int_to_str[i] for i in token_ids])
        return re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
```

#### ➕ Features:

* `encode()` = text → token IDs
* `decode()` = token IDs → text (cleans punctuation spacing)

---

### 5. ❌ Problem: Unknown Tokens

If a token is not in the vocabulary, the encoder will raise an error.

> E.g., if `“hello”` didn’t exist in the training data, it cannot be mapped.

---

### 6. ✅ Enhanced Tokeniser (V2) with Special Tokens

To solve unknown tokens, we add:

| Token   | Purpose                        |
| ------- | ------------------------------ |
| `<unk>` | Represents unknown words       |
| `<eot>` | Represents end of a text chunk |

```python
tokens.extend(["<unk>", "<eot>"])
vocab_dict = {token: idx for idx, token in enumerate(sorted(set(tokens)))}
```

📌 Vocabulary size becomes **1132**

---

#### Updated Tokeniser Class with Fallback

```python
class SimpleTokenizerV2:
    def __init__(self, vocab_dict):
        self.str_to_int = vocab_dict
        self.int_to_str = {i: s for s, i in vocab_dict.items()}

    def encode(self, text):
        tokens = tokenize(text)
        return [self.str_to_int.get(token, self.str_to_int["<unk>"]) for token in tokens]

    def decode(self, token_ids):
        text = " ".join([self.int_to_str[i] for i in token_ids])
        return re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
```

---

### 7. 🏷️ Other Common Special Tokens

| Token   | Purpose                                                       |
| ------- | ------------------------------------------------------------- |
| `<unk>` | Handles unknown/unseen words                                  |
| `<eot>` | Marks end of a text segment (especially for multiple sources) |
| `<bos>` | Beginning of sequence                                         |
| `<eos>` | End of sequence (used in autoregressive generation)           |
| `<pad>` | Used for padding during batching                              |

> GPT models mainly use `<eot>`; they **do not use `<unk>` or `<pad>`**

---

### 8. ⚠️ Limitations of Word-Level Tokenisers

* Treats each word and punctuation as a token
* Doesn’t handle unknown words robustly
* Can’t split rare/compound/complex words efficiently

---

### 9. ✂️ Introduction to Byte Pair Encoding (BPE)

To overcome limitations, GPT and most LLMs use **Byte Pair Encoding (BPE)**.

#### 🔹 BPE Features:

* Breaks words into **subword units**
* Reduces vocabulary size
* Eliminates need for `<unk>` tokens

Example:

```text
"playable" → ["play", "able"]
"newword" → ["new", "word"]
```

---

### ✅ Summary

* Tokenisation is a **foundational step** in LLMs.
* It includes:

  * Splitting text into tokens
  * Mapping tokens to unique IDs
* Special tokens like `<unk>`, `<eot>` are essential for robustness.
* While basic tokenisers are helpful for learning, **real-world models use BPE**.

> Next: A deep dive into **Byte Pair Encoding (BPE)** as used in GPT.

---
## 🧠 GPT Tokenisation using Byte Pair Encoding (BPE)

### 1. 🔎 Introduction to Tokenisation in LLMs

Tokenisation is a fundamental step in preparing text data for LLMs. It converts raw text into smaller units called **tokens**, which are then transformed into **numerical token IDs** for processing by neural networks.

#### 🔁 Types of Tokenisation Algorithms

##### 🔹 Word-based Tokenisation
Each word becomes a unique token.

**Example**:  
`"My hobby is playing cricket"` → `"My"`, `"hobby"`, `"is"`, `"playing"`, `"cricket"`

**Drawbacks**:
- ❌ **Out-of-Vocabulary (OOV)**: Unseen words cause errors.
- ❌ **Large Vocabulary**: English has ~170K–200K words.
- ❌ **Loss of Semantic Similarity**: Words like `"boy"` and `"boys"` are treated as totally distinct.

---

##### 🔹 Character-based Tokenisation
Each character is a token.

**Example**:  
`"My hobby"` → `"M"`, `"y"`, `" "`, `"h"`, `"o"`, `"b"`, `"b"`, `"y"`

**Advantages**:
- ✅ No OOV issue — small, fixed vocabulary
- ✅ Memory-efficient

**Problems**:
- ❌ Loses word-level meaning
- ❌ Leads to long token sequences (e.g., `"dinosaur"` → 8 tokens)

---

### 2. 🧩 Subword-based Tokenisation: The Best of Both Worlds

**Subword tokenisation**, such as **Byte Pair Encoding (BPE)**, overcomes the issues of word- and character-level methods.

#### 🔑 Core Principles of Subword Tokenisation
- **Keep common words whole**: `"boy"` remains `"boy"`
- **Break rare or complex words into sub-parts**:  
  `"boys"` → `"boy"`, `"s"`  
  `"tokenization"` → `"token"`, `"ization"`

#### ✅ Advantages of BPE
- ✅ **Captures roots**: Groups `"token"`, `"tokens"`, `"tokenizing"`
- ✅ **Handles OOV words** by breaking them into smaller known parts
- ✅ **Manages vocabulary size** efficiently
- ✅ **Shorter sequences** than character-based tokenisation
- ✅ **Learns suffixes/prefixes** like `"isation"`, improving understanding

---
### 3. ⚙️ Byte Pair Encoding (BPE) Algorithm

BPE, originally a **data compression algorithm introduced in 1994**, has been adapted for tokenisation in LLMs. It addresses the limitations of word- and character-level tokenisation by iteratively merging the most common adjacent character or subword pairs.

#### 🧱 Original Data Compression Algorithm

**Process**:  
- Find most frequent pair of adjacent bytes/characters
- Replace them with a new symbol not in the data
- Repeat until no pair occurs more than once

**Example**:

```

Initial data:        aaabdaabac
Most common pair:    aa (4 times)
→ Replace 'aa' with 'z'

New data:            zabdzaac
Next common pair:    ab (2 times)
→ Replace 'ab' with 'y'

New data:            zydzac
Stop:                No pair occurs more than once.

````

This illustrates how BPE compresses data by merging frequent byte pairs.
---

#### ✍️ NLP BPE Example (From Lecture):
Using a sample dataset of word frequencies:

```python
{
  "old": 7,
  "older": 3,
  "finest": 9,
  "lowest": 4
}
````

These frequency counts represent how often each word appears in the training corpus and directly influence which character pairs are merged first.

---

##### Step-by-Step:

1. **Preprocess** each word by adding `/w` to mark word end:

   ```
   "old"     → ["o", "l", "d", "/w"]
   "older"   → ["o", "l", "d", "e", "r", "/w"]
   "finest"  → ["f", "i", "n", "e", "s", "t", "/w"]
   "lowest"  → ["l", "o", "w", "e", "s", "t", "/w"]
   ```

2. **Frequency Table** (weighted by word occurrences):

   * `"e"` appears in `"finest"` (9×) and `"lowest"` (4×) → total = **13**
   * `"s"` appears in same → **13**
   * `"e + s"` = **13 times**
   * `"s + t"` = **13 times**
   * `"est + /w"` = **13 times**
   * `"o + l"` appears in `"old"` (7×) and `"older"` (3×) → total = **10**
   * `"l + d"` also = **10 times**

> These pair frequencies are not absolute character counts — they're based on how often the full word appears and how often the pair occurs within that word.

---

##### 3. **Iterative Merging**:

Merges proceed from most frequent pairs to least:

* `"e"` + `"s"` → `"es"`  (13 times)
* `"es"` + `"t"` → `"est"`  (13 times)
* `"est"` + `"/w"` → `"est/w"`  (13 times)
* `"o"` + `"l"` → `"ol"`  (10 times)
* `"ol"` + `"d"` → `"old"`  (10 times)

---

This approach learns that:

* `"est"` is a common suffix in `"finest"` and `"lowest"`
* `"old"` is a root shared by `"old"` and `"older"`

These are now **merged subword tokens**, helping the model understand that these words share structure and meaning.

---

##### 4. **Stopping Criteria**:

The merging process continues until:

* A desired **token vocabulary size** is reached (e.g., \~**50,000–57,000** tokens for GPT-2/GPT-3)
* Or no pair occurs more than once

This controls the **granularity** and **capacity** of the model’s vocabulary.

---

### 4. 🛠️ Practical Implementation Using `tiktoken`

OpenAI’s models like GPT-2 and GPT-3 use BPE via the open-source **`tiktoken`** library.

#### 📦 Install
```bash
pip install tiktoken
````

#### 🔧 Initial Setup

```python
import tiktoken

# Load GPT-2 compatible BPE tokenizer
tokenizer = tiktoken.get_encoding("gpt2")
```

---

#### ▶️ Example 1: Encode and Decode with BPE

```python
text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
    "of someunknownPlace."
)

integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(integers)
# Output: [15496, 11, 466, 345, 588, 8887, 30, 220, 50256, 554, 262, 4252, 18250, 8812]

decoded_text = tokenizer.decode(integers)
print(decoded_text)
# Output: Hello, do you like tea? <|endoftext|> In the sunlit terracesof someunknownPlac
```

---

#### ▶️ Example 2: Handling Unknown Words Gracefully

```python
encoded = tokenizer.encode("Akwirw ier")
print(encoded)
# Output: [33901, 86, 343, 86, 220, 959]

decoded = tokenizer.decode(encoded)
print(decoded)
# Output: Akwirw ier
```

✅ Even made-up or rare words are broken into smaller known subwords — **no crash or error**.

---

#### 🧾 Vocabulary Size in GPT

* GPT-2/GPT-3 use **50,257 tokens**.
* Token ID `50256` = `<|endoftext|>`

> This is much smaller and more efficient than full word-based vocabularies with over 170K entries.

---

### 5. ✅ Summary: Why BPE Works So Well

Byte Pair Encoding offers a powerful, efficient solution for LLM tokenisation. It:

* 🧠 Preserves **word meaning**
* 💬 Identifies **roots and affixes**
* 📉 Keeps **token sequences short**
* 📦 Keeps **vocabulary size manageable**
* 🧩 Handles **unknown words robustly**

> BPE is the engine that powers tokenisation in GPT — making it efficient, flexible, and scalable.

---

### 📚 Acknowledgement

> 🧠 *This content is inspired by Dr. Raj Dandekar’s YouTube playlist on Large Language Models.*
> It has been adapted for educational purposes into a markdown-friendly format for easy reference and learning.

---
