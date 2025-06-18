# Large Language Models (LLM) From Scratch

> _This content is inspired by Dr. Raj Dandekar’s YouTube playlist on Large Language Models. It will be updated as and when more topics are covered._

---

## 📑 Table of Contents

1. [📘 Large Language Models (LLM) Basics](#-large-language-models-llm-basics)
2. [🧪 Pretraining LLMs vs Finetuning LLMs](#-pretraining-llms-vs-finetuning-llms)
3. [🤖 Understanding Transformers and LLMs](#-understanding-transformers-and-llms)

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

### 📚 Acknowledgement

> 🧠 *This content is inspired by Dr. Raj Dandekar’s YouTube playlist on Large Language Models.*
> It has been adapted for educational purposes into a markdown-friendly format for easy reference and learning.

---
