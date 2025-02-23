
Course : [Large Language Models (LLMs) Concepts](https://app.datacamp.com/learn/courses/large-language-models-llms-concepts)

------------------------------------------------------------------------------------------------

# Chapter 1 : Introduction to Large Language Models (LLM)
 
------------------------------------------------------------------------------------------------

# The rise of LLMs in the AI landscape

## Rapid developments in AI

AI is everywhere - in facial recognition systems that analyze and identify our facial features to unlock smartphones or other devices and services; or the self-driving car, another fascinating application of AI algorithms that has revolutionized the transportation industry.

## AI-powered recommendations

It assists us with various tasks, from recommending movies to suggesting songs on streaming platforms. AI algorithms analyze user preferences, history, and behavior data to generate personalized recommendations that help users discover the most relevant content.

## AI and data-driven tasks

n addition to personalization, AI has performed well in data-driven tasks such as sentiment analysis, fraud detection, and more. However, it traditionally lacked the ability to understand context, respond to open-ended questions, and generate human-like responses in conversation. Recent developments in language-based AI models have led to a disruptive breakthrough. These models, called large language models, can process language as humans do.

## The AI landscape

Before we discuss large language models, let's break down AI and the fields it includes from largest to smallest.

Let's start with Machine Learning, a subfield of Artificial Intelligence that enables models to learn patterns from data without explicit instructions.

A subset of Machine Learning is Deep Learning, which can recognize complex patterns like those found in computer vision and self-driving cars

Natural Language Processing (NLP) utilizes machine learning techniques, among others, to understand and process human language by computers.

## Definition of LLMs

Large Language Models, which we call LLMs, use Deep Learning techniques to perform a variety of Natural Language Processing (NLP) tasks such as text classification, summarization, generation, and more.

LLMs are called "large" because they require a large amount of training data and resources to work.

They are powerful in processing and analyzing human language data. They have set new benchmarks in various NLP tasks, outperforming their predecessors and opening up new possibilities in AI.

"Models" corresponds to the capability of learning complex patterns using data. In the case of LLMs, the data is text from the internet.

## The defining moment

The LLM is considered the defining moment, or the iPhone moment, in the history of AI. It has been the talk of many conversations - technical and non-technical alike.

## Popular language generators

The GPT series by OpenAI is one of the most popular language models among the family of LLMs, primarily because of its advanced ability to engage in rich human interactions. There are other language generators in the market, and new ones are added every day.

## Applications

LLMs can perform a wide range of tasks, such as sentiment analysis, identifying themes, translating text or speech, and even generating code! They are also used to predict the next words in a text based on what the user has already typed. Isn't that amazing?





# Real-world applications

Let's explore some real-world applications of these powerful AI tools across various industries to understand the business opportunities and benefits of leveraging LLMs.

## Business opportunities

Large language models have numerous applications and benefits across various industries, helping automate tasks, improve efficiency, create revenue streams, and enable new capabilities. The possibilities are endless. Businesses constantly seek new and innovative ways to improve their products and services using LLMs. We will examine how LLMs have transformed the finance, healthcare, and education industries.

## Finance industry

Let's start with the finance industry. Financial analysis of a company can be complex and may include processing unstructured text such as investment outlooks, annual reports, news articles, and social media posts. Unstructured data or text refers to data that lacks a pre-defined format and is typically presented in free-form text. LLMs can analyze such data to generate valuable insights into market trends, manage investments, and identify new investment opportunities.

## Healthcare

Now let's look at the healthcare industry. Analyzing health records is important for giving personalized recommendations to provide quality healthcare. But, much of the information is in doctors' notes, which can be hard to understand because they use jargon and abbreviations. Furthermore, domain-specific knowledge and varying writing styles among practitioners add to the challenges of interpreting this critical information effectively. Processing such varied sources of text data and understanding complex acronyms makes it difficult to have a general system to describe any patient files.

But not anymore. LLMs have made it possible. LLMs can analyze large amounts of patient data, such as medical records, health check-up results, imaging reports, and more, to provide personalized treatment recommendations. Patient data is private and personal information, so anyone using LLMs this way must adhere to privacy laws and regulations.

## Education

Our third and final example is the transformation in the education industry. Have you ever wished for a tutor who could personalize the coaching style based on the style and expectations of the learner? Well, here is some good news. Education companies and schools are integrating LLMs into their platforms to provide interactive learning experiences to learners. The learners can ask questions, receive guidance, and discuss their ideas with an AI-powered tutor. Moreover, the tutor can adapt its teaching style based on the learner's conceptual understanding.

LLMs can be used for text generation, such as customized learning materials, which include explanations, examples, and exercises based on a learner's current knowledge and progress. We can observe in this example that the model can adapt the explanation of the same concept for a child and an astronomy expert. The response style and the level of detail show how an experienced professor might have approached the explanation

## Defining multimodal

Now that we understand how different industries can apply LLMs let's discuss the multimodal applications of LLMs. But what do we mean by multimodal? Multi means many, and modal means modes or types. These models can process and generate information across different data types, such as text, audio, video, and images, in contrast to non-multimodal, which work with only one of the modes, such as text only.

## Visual question answering

Visual question answering is one such multimodal application. LLMs can generate meaningful and contextually relevant answers to questions about visual content, such as identifying objects, understanding relationships between them, and describing scenes. For example, we can see that the model recognizes the zebra image and also responds with additional information, such as making a joke. This multimodal application processed both image and text data to generate responses.





# Challenges of language modeling

## Sequence matters!

Modeling a language requires understanding the sequential nature of text because placing even one word differently can change the meaning of the sentence completely. Take a look at these sentences: "I only follow a healthy lifestyle" and "Only I follow a healthy lifestyle". In this example, the word "only" is used in different positions leading to different meanings.

## Context modeling
There is more to it than just the order of the words. Language is highly contextual, meaning the same word can have different meanings depending on the context in which it is used. For example, the word "run" can have different meanings in different contexts,

such as "to jog,"

"to manage or organize,"

or "to operate a machine."


To accurately model language, language models must analyze and interpret the surrounding words, phrases, and sentences to identify the most likely meaning of a given word. In the first example, the model references the word "marathon" to understand that "run" implies jogging. In the second example, it utilizes the context from the word "organization" to understand that "run" here means "to manage". In the third example, the word "machine" indicates to the model that "run" here means "to operate".

## Long-range dependency

Consider the sentence: "The book that the young girl, who had just returned from her vacation, carefully placed on the shelf was quite heavy." To understand the link between the book and its weight, the model needs to correctly connect these words even if they are far apart in the text. This requires the model to recognize and maintain a long-range dependency between two distant parts of the sentence. In this case, the model needs to keep track of the words "book" and "was quite heavy" to understand the sentence. This can be challenging for traditional language models.

## Single-task learning

Traditional models are trained for each specific task, known as single-task learning. For example, one model would be trained for individual tasks like question-answering, text summarization, and language translation. This approach requires significant resources and time as each model has to be developed and trained independently. Additionally, these models are limited in their ability to incorporate multiple data modalities, such as text, images, and other data types. This means they are less flexible and not as powerful compared to modern LLMs.

## Multi-task learning

With the development of LLMs, multi-task learning has become possible. This involves training a model to perform multiple related tasks simultaneously instead of training separate models for each task. Training a model on multiple related tasks can improve its ability to predict using new and unseen data, but may come at the expense of accuracy and efficiency. Note that multi-task learning can decrease the training data needed for each individual task by allowing the model to learn from shared data across the tasks.





------------------------------------------------------------------------------------------------

# Chapter 2 : Building Blocks of LLMs

------------------------------------------------------------------------------------------------





# Novelty of LLMs

## Using text data

Recall that LLMs use text data in various ways. Text data used in sentiment analysis, spam classification, and digital assistants is unstructured and can be messy and inconsistent.

## Machines do not understand language!

Besides, computers cannot understand language in the same way that humans do. For example, a computer does not know how to process raw text like "I am a data scientist".

## Need for NLP

This is because they don't read the text as we do. Instead, they read in numbers, which is the language of computers. Natural Language Processing (NLP) techniques address this challenge by converting the text into numerical form, enabling machines to identify patterns and structures. These NLP techniques are the foundations of LLMs.

## Unique capabilities of LLMs

After exploring NLP's role in language learning, let us understand the unique capabilities of LLMs. The novelty of an LLM is the ability to detect linguistic subtleties like irony, humor, pun, sarcasm, intonation, and intent.

## What's your favorite book?

For example, an LLM can provide a human-like response when asked about its favorite book. It may start with a natural response like "Oh, that's a tough one," followed by a personal recommendation, "My all-time favorite book is To Kill a Mockingbird by Harper Lee". To support its choice, it may highlight a key theme and initiate a further exchange by asking, "Have you read it?" making the conversation more natural.

## Linguistic subtleties

Let's explore how LLMs understand linguistic subtleties better than traditional language models. Consider the sarcastic statement, "Oh great, another meeting." The traditional model responds neutrally with "What's the meeting about?" and fails to capture the underlying sarcasm, while an LLM generates "Sounds like you're looking forward to it", a playful and engaging response that matches the sarcastic tone.

## How do LLMs understand

These examples demonstrate the impressive ability of LLMs to understand language. But what makes this possible? As we learned earlier, LLMs are considered "large" because they are trained on vast data. Another key factor contributing to their "largeness" is parameters. Parameters represent the patterns and rules learned from training data. More parameters allow for capturing more complex patterns, resulting in sophisticated and accurate responses.

## Parameters

The concept is similar to building with Lego bricks, where a few bricks only allow for a simple structure, while a larger number of bricks can create detailed structures.

## Emergence of new capabilities

These massive parameters also give rise to LLMs' emergent capabilities, unique to large-scale models like LLMs and not found in smaller ones. Scale is determined by two factors: the volume of training data and the number of model parameters. As the scale increases, the model's performance can dramatically improve beyond a certain threshold, resulting in a phase transition and a sudden emergence of new capabilities, such as music and poetry creation, code generation, and medical diagnosis.

## Building blocks of LLMs

To reach this threshold, LLMs and their parameters undergo a training process - text pre-processing, text representation, pre-training, fine-tuning, and advanced fine-tuning. We will cover each of these in the upcoming videos.





# Generalized overview of NLP

## Text pre-processing

Text pre-processing transforms raw text data into a standardized format and involves several steps, including tokenization, stop word removal, and lemmatization. Note that these pre-processing steps are independent and can be done in a different order depending on the task. Let's learn about each one.

## Tokenization

Tokenization splits the text into words, also called tokens. Consider the sentence: “Working with natural language processing techniques is tricky”. It is broken into words as – ["Working", "with", "natural", "language", "processing", "techniques", "is", "tricky", "."]. Note that punctuation is also a token. Square brackets represent a list. As the series of words are now considered a list, they are no longer a sentence anymore.

## Stop word removal

Sometimes, when working with text data, we may come across frequently used words, such as "with" or "is," that don't add much meaning to the text, known as stop words. These additional words are eliminated through a step called stop word removal to identify the most important parts of the sentence.

## Lemmatization

Often, we may have slightly different words that mean the same thing in the context of the sentence. This means we can group these words together. This process of reducing words to their base form is known as lemmatization. For example, "talking", "talked", and "talk" would be mapped to the root word "talk".

## Text representation

Text representation techniques help convert preprocessed text into a numerical form. There are different ways of doing this, but we will focus on bag-of-words and word embeddings.

## Bag-of-words

The bag-of-words approach involves converting the text into a matrix of word counts. Consider the two sentences: “The cat chased the mouse swiftly” and “The mouse chased the cat” After removing the stop words, the bag-of-words technique creates a list of all the unique words and their count, such as “cat”, “chased”, “mouse”, and "swiftly". Note that in the first sentence, the last digit is one, which corresponds to the word "swiftly". However, since sentence two does not contain this word, it is represented with the digit zero.

## Limitations of bag-of-words

The bag-of-words method has its limitations - it fails to capture the meaning and context of words, leading to incorrect interpretations of a text. For example, these sentences are similar, but their meaning is the opposite. Further, it treats related words, such as "cat" and "mouse," as separate and independent, failing to capture their semantic relationship.

## Word embeddings

Word embeddings address these limitations by capturing semantic meanings of words and representing them as numbers, allowing for similar words to have similar representations. For example, the cat is a predator and the mouse is prey. Word embeddings will convert these into numbers, with higher numbers indicating a stronger meaning. So the word "cat" becomes minus 0-point-9, 0-point-9, and 0-point-9. Plant, furry, and carnivore are features learned by the model using the training data. In reality, we won't see these labels but instead, see a word defined as a list of numbers. This technique allows us to represent similar relationships between other words like tiger and deer, and eagle and rabbit.

## Machine-readable form

To recap, several techniques such as tokenization, stop word removal, and lemmatization are used to pre-process text data.

Which is then transformed into a numerical format using techniques like bag-of-words and word embeddings, enabling it to be used by LLMs.





# Fine-tuning

In this video, we will learn the key challenges of pre-training an LLM and examine how fine-tuning addresses those concerns.

## Where are we?

Not everyone needs to train an LLM from scratch, as pre-trained models from industry leaders can be fine-tuned for specific tasks. Hence, we will first explore fine-tuning, while pre-training will be covered in the next chapter.

## Fine-tuning analogy

Pre-training can be thought of as similar to how children learn to speak a language by observing their surroundings at home and school. As they enter college and choose to specialize in a specific area, such as medicine, they fine-tune their language understanding based on specific vocabulary and language patterns unique to that domain. This allows them to communicate more effectively with others in their chosen field.

## "Largeness" challenges

Fine-tuning is an effective approach used to help LLMs overcome certain challenges. We have discussed the scale and uses of LLMs in various NLP applications, but the "largeness" of these models also presents several challenges. Building these models requires powerful computers and specialized infrastructure due to the massive amounts of data and computational resources involved. Additionally, efficient model training methods and the availability of high-quality training data are essential for optimal model performance.

## Computing power

One major challenge is the high computational cost of training and deploying LLMs. The sheer size of these models requires a significant amount of memory, processing power, and infrastructure which is quite expensive and difficult to manage. An LLM may require a few hundred thousand Central Processing Units (CPUs) and tens of thousands of Graphic Processing Units (GPUs) compared to 4-8 CPUs and 0-2 GPUs in a personal computer. This level of computing power requires large-scale infrastructure, which can be extremely expensive to set up and maintain.

## Efficient model training

Training an LLM is another key challenge, as it requires significant training time, often weeks or even months. Efficient model training can lead to faster training times and reduce costs. Training an LLM might take as much as 355 years of processing on a single GPU.

## Data availability

Another challenge is the need for high-quality training data to accurately learn the complexities and subtleties of language. For instance, an LLM is trained on a few hundred gigabytes (GBs) of text data equivalent to more than a million books. That's a massive amount of data to process!

## Overcoming the challenges

Fine-tuning addresses some of these challenges by adapting a pre-trained model for specific tasks. Pre-trained language models typically learn from large, general-purpose datasets and are not optimized for specific tasks. However, because of the general language structure and flow they learn, they might be an ideal candidate for fine-tuning to a specific problem or dataset. We will explore the pre-training process in more detail in the next chapter.

## Fine-tuning vs. Pre-training

Fine-tuning is more effective since it can help a model learn, or be trained, using a single CPU and GPU, while pre-training may require thousands of CPUs and GPUs to train efficiently. Additionally, fine-tuning can take hours or days, while training a model from scratch may take weeks or months. Furthermore, fine-tuning requires only a small amount of data, typically ranging from a few hundred megabytes to a few gigabytes, compared to hundreds of gigabytes as are necessary for pre-training.





# Learning techniques

In this video, we will examine how to handle data availability when creating an LLM.

## Where are we?

Here, we will discuss the learning techniques used in fine-tuning a pre-trained LLM.

## Getting beyond data constraints

Fine-tuning involves training a pre-trained model on a smaller, task-specific labeled dataset to improve performance. But what if little to no labeled data is available to learn a specific task or domain? This is where zero-shot, few-shot, and multi-shot learning comes in, collectively called N-shot learning techniques.

## Transfer learning

These techniques are all part of transfer learning. So, what is transfer learning? It involves training a model on one task and applying the learned knowledge to a different but related task. For example, the skills acquired during piano lessons, such as reading musical notes, understanding rhythm, and grasping musical concepts, can be quickly transferred when learning to play the guitar. In the case of LLMs, the pre-trained language model is fine-tuned on a new task with little to no task-specific training data (zero-shot or few-shot) or with more training data (multi-shot).

## Zero-shot learning

Zero-shot learning allows LLMs to perform a task it has not been explicitly trained on. It uses its understanding of language and context to transfer its knowledge to the new task. Suppose a child has only seen pictures of horses and is asked to identify a zebra with additional information that it looks like a striped horse. In that case, they can correctly identify it without seeing specific examples of zebras. It demonstrates how LLMs use zero-shot learning to quickly learn new skills and generalize knowledge to new situations without using any examples.

## Few-shot learning

On the other hand, few-shot learning allows the model to learn a new task with very few examples. This is achieved using the knowledge the model has gained from previous tasks, allowing it to generalize to new tasks with only a few examples. For example, a student attends lectures and takes notes but doesn't study extra for exams. On exam day, they encounter a new question similar to the one taught in the class and can answer it correctly by relying on prior knowledge and experience. When the number of examples used for fine-tuning is only one, it is called one-shot learning.

## Multi-shot learning

Multi-shot learning is similar to few-shot learning but more examples are required for the model to learn a new task. The model uses the knowledge learned from previous tasks, along with more examples of the new task, to learn and generalize to new tasks. Let's take an example of recognizing different breeds of dogs. If we show the model a few pictures of a Golden Retriever, it can quickly learn to recognize the breed and then generalize this knowledge to similar breeds with just a few more examples, such as Labrador Retriever. This approach saves time and effort in collecting and labeling a large dataset for each breed while still achieving good accuracy in recognizing different dog breeds.

## Building blocks so far

So far, we have covered a lot of ground. We learned how data is prepared for computers to understand language and discussed how fine-tuning is an effective approach for overcoming the challenges of building LLMs. We also explored N-shot learning techniques for dealing with the lack of data. In the next chapter, we will dive deeper into how LLM models are pre-trained.





------------------------------------------------------------------------------------------------

# Chapter 3 : Training Methodology and Techniques

------------------------------------------------------------------------------------------------





# Building blocks to train LLMs

This video will focus on two popular pre-training techniques to build LLMs - next word prediction and masked language modeling.

## Where are we?

These pre-training techniques form the foundation for many state-of-the-art language models. Recall that we prioritized discussing fine-tuning over pre-training because many organizations opt to fine-tune existing pre-trained models for their specific tasks rather than building a pre-trained model from scratch.

## Generative pre-training

LLMs are typically trained using a technique called generative pre-training. This technique involves providing the model with a dataset of text tokens and training it to predict the tokens within it. Two commonly used styles of generative pre-training are next word prediction and masked language modeling.

## Next word prediction

Let's start with the next word prediction. It is a supervised learning technique that trains the model on the input data and its corresponding output. Remember, supervised learning uses labeled data to classify or predict new data. Similarly, a next word prediction model is used to train a language model to predict the next word in a sentence, given the context of the words before it. The model learns to generate coherent text by capturing the dependencies between words in the larger context. During training, the model is presented with pairs of input and output examples.

## Training data for next word prediction

For example, from the sentence "The quick brown fox jumps over the lazy dog", we can create input-output pairs for the model to learn from. During training, each generated output is added to the input for the next pair, allowing the model to predict the next output. For example, the output "fox" was generated for the input, "The quick brown" in the first pair. Now, this output "fox" gets added to the input of the second pair to become "The quick brown fox". The model takes this second input to predict "jumps" based on the context of the previous input-output pair. This is just one example. An LLM is typically trained on a large amount of such text data.

## Which word relates more with pizza?

The more examples it sees, the better it predicts the next word. Once trained, we can use the model to generate new sentences one word at a time. For example, if we prompt it with "I love to eat pizza with blank", it is more likely to generate "cheese" instead of any other word like oregano, coffee, or ketchup. The model has learned from the training data that "cheese" occurs more often with pizza than anything else. Note that the probability of occurrence of the words here is a hypothetical example and not based on any specific data.

## Masked language modeling

The second style of generative pre-training we will learn about is masked language modeling which involves training a model to predict a masked word that is selectively hidden in a sentence. For instance, if we mask the word "brown" in "The quick brown fox jumps over the lazy dog.", the sentence becomes "The quick [MASK] fox jumps over the lazy dog." During training, the model receives both the original and masked texts as input. The model's objective is to correctly predict the missing word between "quick" and "fox". Even though the masked word could be any color, the model has learned that it's "brown" based on the training data.





# Introducing the transformer

## Where are we?

Transformers are part of pre-training and enhance the techniques we have already learned about.

## What is a transformer?

It all started with the release of the “Attention Is All You Need” research paper that changed how language modeling is done today. The transformer architecture emphasizes long-range relationships between words in a sentence to generate accurate and coherent text. It has four essential components: pre-processing, positional encoding, encoders, and decoders.

## Inside the transformer

Let's consider an input text, "Jane, who lives in New York and works as a software". The transformer pre-processes input text, converting it to numbers and incorporating position references. The encoder uses this information to encode the sentence, which the decoder then uses to predict subsequent words. The predicted word is added to the input, and the process continues until the final output completes the input sentence. Here, the final output is "engineer, loves exploring new restaurants in the city". Let's walk through these steps.

## Transformers are like an orchestra

Imagine the transformer as an orchestra.

## Text pre-processing and representation

The first component is text pre-processing and representation where the transformer breaks down the sentences into individual tokens, like a composer separating the music into individual notes. Recall that tokenization breaks sentences into tokens and that stop word removal and lemmatization are also text pre-processing techniques. These tokens then need to be presented in numerical form using word embeddings, a text representation technique. This is similar to sheet music providing a set of instructions to musicians to interpret and play 

## Positional encoding

The second component is positional encoding, which provides information about the position of words in a sequence, helping a transformer tie together distant words. This is similar to understanding the relationships between distant notes that create a coherent piece of music.

## Encoders

The third component, encoders, includes the attention mechanism and neural network. The attention mechanism directs attention to specific words and their relationships. In music, this is similar to musicians adjusting their volume in specific sections. We'll explore this mechanism more in the next video. Recall that neural networks are algorithms inspired by the human brain. The different layers of neural networks process specific features of the input data to interpret complex patterns and pass them to the next layer, just as each musician contributes to the final musical piece.

## Decoders

The decoders, the fourth component, also use attention and neural networks to process the encoded input and generate the final output. This is similar to how individual musicians combine their knowledge as an orchestra to create a cohesive and meaningful performance. Great, so we understand how transformers work. Let's check out what makes them special.

## Transformers and long-range dependencies

Recall that long-range dependencies require capturing relationships between distant words in a sentence - which can be challenging to model. The transformer's attention mechanism overcomes this limitation by focusing on different parts of the input. Going back to our previous example: "Jane, who lives in New York and works as a software engineer, loves exploring new restaurants in the city", LLMs can attend to the relationship between the distant words - "Jane" and "loves exploring new restaurants", leading to better contextual understanding.

## Processes multiple parts simultaneously

When handling language, traditional language models are sequential, meaning they process one word at a time. Transformers are an improvement in this area, because they focus on multiple parts of the input text simultaneously, speeding up the process of understanding and generating text. For example, in the sentence "The cat sat on the mat," transformers can process "cat," "sat," "on," "the," and "mat" at the same time.





# Attention mechanisms

In this video, we will dive into how the attention mechanism works, exploring its power to capture relationships between words and improve language modeling.

## Attention mechanisms

Attention mechanisms help language models understand complex structures and represent text more effectively by focusing on important words and their relationships. To better understand how attention works, consider reading a mystery book. As you would focus on clues while ignoring less important content, attention enables models to identify and concentrate on crucial input data.

## Self-attention and multi-head attention

Now that we better understand attention as a concept let's explore its two primary types - self-attention and multi-head attention. Self-attention weighs the importance of each word in a sentence based on the context to capture long-range dependencies. Multi-head attention takes self-attention to the next level by splitting the input into multiple "heads". Each head focuses on different aspects of the relationships between words, allowing the model to learn a richer representation of the text.

## Attention in a party

Let's look at an example, starting with attention and later extending it to differentiate between self and multi-head attention. In a group conversation at a party, it is common to selectively pay attention to the most relevant speakers to understand the topic being discussed. By filtering out background noise or less important comments, individuals can focus on the key points of the conversation and understand what is being discussed.

## Party continues

Self-attention can be compared to focusing on each person's words in the group conversation and evaluating their relevance in relation to other people's words. This technique enables the model to weigh each speaker's input and combine them to form a more comprehensive understanding of the conversation. In contrast, multi-head attention involves splitting attention into multiple "channels" that simultaneously focus on different aspects of the conversation. For instance, one channel may concentrate on the speakers' emotions, another on the primary topic, and a third on related side topics. Each aspect is processed independently, and the resulting understandings are merged to gain a holistic perspective of the conversation.

## Multi-head attention advantages

Let's review this using text. Consider the following sentence: "The boy went to the store to buy some groceries, and he found a discount on his favorite cereal." The model pays more attention to relevant words such as "boy", "store", "groceries", and "discount" to grasp the idea that the boy found a discount on groceries at the store. When using self-attention, the model might weigh the connection between "boy" and "he" recognizing that they refer to the same person. It also identifies the connection between "groceries" and "cereal" as related items within the store. Multi-head attention is like having multiple self-attention mechanisms working simultaneously. It allows the model to split its focus into multiple channels where one channel might focus on the main character ("boy"), another on what he's doing ("went to the store," "found a discount"), and a third on the things involved ("groceries," "cereal"). These two attention mechanisms work together to give the model a comprehensive understanding of the sentence.





# Advanced fine-tuning

## Where are we?

Advanced fine-tuning is our final building block for LLMs. It's time to understand how everything comes together to give rise to these colossal models.

## Reinforcement Learning through Human Feedback

So far, we have learned the two-stage process of training an LLM - pre-training and fine-tuning. In this video, we learn about the third phase of LLM training, the Reinforcement Learning through Human Feedback (RLHF) technique. But first, let's quickly revisit pre-training and fine-tuning steps.

## Pre-training

Let's recall that LLMs are pre-trained on large amounts of text data from diverse sources, like websites, books, and articles, using transformer architecture. Its primary goal is to learn general language patterns, grammar, and facts. During this stage, the model learns to predict the next word or missing word using next-word prediction or masked language modeling techniques.

## Fine-tuning

After pre-training, the model is fine-tuned using N-shot techniques (such as zero, few, and multi-shot) on small labeled datasets to learn specific tasks.

## But, why RLHF?

So, why do we need a third technique, RLHF? The concern is that the large general-purpose training data may contain noise, errors, and inconsistencies, reducing its accuracy in specific tasks. For example, when a model is trained on data from online discussion forums, it will have a mix of unvalidated opinions and facts. The model treats this training data as the truth, therefore reducing the accuracy. RLHF introduces an external expert to validate the data and avoid these inaccuracies.

## Starts with the need to fine-tune

While pre-training enables the model to learn underlying language patterns, it may not capture the complexities of language in these specific contexts. During the fine-tuning stage the model's performance can improve using quality labeled data for specific tasks. This is where techniques like RLHF come into play. It is an advanced fine-tuning technique that unlocks the true potential of language models by gathering human feedback.

## Simplifying RLHF

In this approach, the model generates output, which is then reviewed by a human who provides feedback on how well the model performed. The model is then updated based on the feedback to improve its performance over time. Let's break this down into three steps. First, the model generates multiple responses to a given question or prompt based on what it has learned from reading lots of text.

## Enters human expert

Next, a human expert, such as a language teacher or someone who knows the topic well, is presented with these different responses generated by the model. The expert ranks the responses according to their quality, such as their accuracy, relevance, and coherence. This ranking process provides valuable information to the model about which responses are better or worse.

## Time for feedback

Finally, the model learns from the expert's ranking of responses, trying to understand how to generate better responses in the future that align with the expert's preferences. The model continues to generate responses, receive rankings from the expert, and learn from the feedback, improving its ability to provide helpful and accurate information over time.

## Recap

In summary, pre-training an LLM captures general language knowledge, followed by fine-tuning on specific tasks, which is further enhanced with RLHF techniques to incorporate human feedback and preferences. This combination of training methods allows the model to become highly effective at understanding and generating human-like text for various applications.

After the advanced fine-tuning step of RLHF, we completed the training process.




------------------------------------------------------------------------------------------------

# Chapter 4 : Concerns and Considerations

------------------------------------------------------------------------------------------------





# Data concerns and considerations

So far, we have discovered how large language models (LLMs) are changing the AI landscape, especially in how we use language, and summarized how they are constructed.

## Data considerations

In this video, we will examine the data considerations to build these large models, such as data volume and compute power, quality of data, labeling status, bias in data, and data privacy.

## Data volume and compute power

We will discuss them one by one, starting with data volume and compute power. Think about how a child learns to talk. They need to hear lots of words, many times over, to start talking. Training LLMs is similar - they need a ton of data to learn language patterns and structures. Recall that an LLM may need 570 GB of data to train, which is equivalent to 1-point-3 million books.

## Data volume and compute power

The computing power needed to process this sheer magnitude of data is extensive. Think of the extent of energy consumption that goes into its making, something we will discuss in the next video. To give an estimate of scale, training one such model can cost millions of dollars worth of computational resources.

## Data quality

The next factor is high-quality data, which is crucial to train an LLM. Accurate data leads to better learning and improved generated response quality, building trust in its outputs. Let's go back to the child learning to talk. They will learn what they have heard, even if it's gibberish. The same goes for LLMs. They will produce low-quality outputs if we train them with data full of mistakes or poor grammar.

## Labeled data

Ensuring correct data labeling is crucial for training LLMs as it enables the model to learn from accurate examples, generalize patterns, and generate accurate responses. However, this process can be labor-intensive due to the large amount of data. For example, when training an LLM to categorize news articles like 'Sports', 'Politics', or 'Technology', assigning the correct label requires significant human effort. Misclassifications, or errors, occur when articles are assigned incorrect labels, impacting the model's reliability and performance. To address these errors, the labels are identified and analyzed, leading to iterative model refinement.

## Data bias

Ensuring bias free data is as important as its quality and accuracy for any model including LLMs. Bias occurs when the model's responses reflect societal stereotypes or lack diverse training data, leading to discrimination and unfair outcomes. For example, a sentence starting with "The nurse said that..." might be more likely to be completed with a female pronoun like "she". To address biases, we must actively evaluate the training data for imbalances, promote diversity, and employ bias mitigation techniques, which can include augmenting the dataset with more diverse examples.

## Data privacy

Even if the data has good-quality labels, we also need to consider compliance with data protection and privacy regulations. The data may contain sensitive or personally identifiable information (PII). Privacy is a big deal when it comes to data. Training a model on private data without permission, even if the identifying details are anonymized, can breach privacy, leading to legal consequences, financial penalties, and reputational harm. The relevant permissions need to be obtained so that data privacy laws are followed.





# Ethical and environmental concerns

We know the significant data considerations and privacy issues in developing and using LLMs. Now, we will discuss the ethical concerns and environmental impacts of building LLMs.

## Ethical concerns

Let’s underline the importance of incorporating ethical practices by exploring its different facets, such as transparency risk, accountability risk, and information hazards.

## Transparency risk

We will first discuss transparency risk, which makes it challenging to understand how a model arrived at a particular output. Without transparency, it can be difficult to identify issues like bias, errors, or misuse, making it a black box. For instance, an LLM used to predict disease outcomes must explain the reasoning for making informed treatment decisions.

## Accountability risk

Next is accountability risk, which relates to assigning responsibility for the actions of LLMs. For instance, who is responsible if a model generates incorrect or harmful advice for patients seeking medical attention – the model developer or the company that deployed the model? It's like playing a game and not knowing its rules. We lack transparency and accountability if a mistake is made or harm is caused.

## Information hazards

Information hazards include risks associated with disseminating information that can cause harm. They could manifest in various ways, such as harmful content generation, misinformation spread, malicious use, and toxicity.

Let's illustrate these risks with examples, starting with how a model might generate harmful, offensive, or inappropriate content, either in response to specific prompts or due to biases in the training data. Imagine an LLM that is supposed to write a story about a school. The model writes about bullying instead of a friendly school environment. This could be distressing or harmful to some readers. Another concern surrounding LLMs is that they increase misinformation spread as they can generate text on various topics but cannot verify it. Suppose we ask an LLM, "What's a good diet for losing weight?" to which it might suggest an unsubstantiated diet plan that may be harmful.

Malicious use is the third category of information hazard, where bad actors could use models to generate deceptive content that causes harm. An LLM could be exploited to create convincing yet fabricated news that could manipulate public opinion or lead to potential social unrest. The fourth hazard, toxicity, involves generating inappropriate content when trained on such data or manipulated through malicious prompts. Typical examples include insensitive responses or stereotypes related to gender, race, or ethnicity.

## Environmental concerns

Having explored the ethical concerns, it is important to discuss the ecological footprint left by colossal models such as LLMs. Training LLMs require substantial energy resources, leading to significant environmental impacts, mainly through carbon emissions associated with electricity usage. Recall that training an LLM may require a few hundred thousand CPUs and tens of thousand GPUs, equivalent to thousands of computers filling an entire building and thus consuming electricity proportionately.

## Cooling requires electricity too!

LLMs generate significant heat due to their high computational demands. Think about when our laptops overheat. Now imagine thousands of laptops or computers in a room. This heat production, requiring complex cooling systems, adds to their environmental impact. Balancing this environmental cost with the benefits of LLMs, such as improved data processing and technological advancement, is challenging. One approach is prioritizing renewable energy sources for powering and cooling LLM servers, reducing their carbon footprint. Additionally, advances in energy-efficient computing and cooling technologies can help make LLMs more eco-friendly.





# Where are LLMs heading?

In this video, we will explore the future of LLMs and the exciting research and development happening in this field. By the end, we will better understand the potential advancements and focus areas in the LLM space.

## Journey so far

Before we jump into the future, let's quickly recap what we have learned in our journey with LLMs. We have covered the basics of LLMs, their applications in NLP, and the training processes involving data consideration, ethics, and environmental impact. Now, it's time to look ahead and explore what lies on the horizon for LLMs. 

## Model explainability

Model explainability is a critical aspect of future research. As LLMs become more powerful, it's crucial to understand how they arrive at their outputs. Imagine an LLM planning our road trip. We would be interested to know answers to questions such as, "Why did the model choose this particular route?" and "Why did it suggest these specific spots for me to visit?". Explainability builds trust in the technology and ensures we can identify and correct any biases or errors in the model's decisions.

## Efficiency

In addition to explainability, developers are working on boosting LLMs' computational efficiency for quicker, less power-intensive outputs. Research efforts in model compression and optimization are ongoing, speeding up data processing to save energy and time. These improvements will result in better storage management and lower energy consumption, making LLMs more sustainable and cost-effective. These steps can promote green AI, make LLMs viable on devices with limited resources, and reduce operating costs, improving accessibility and sustainability.

## Unsupervised bias handling

Earlier, we learned that data bias is a key consideration in building LLMs, which can have dire consequences resulting in discrimination. Handling bias in an unsupervised manner is an exciting area of research that explores methods and techniques to detect and mitigate biases automatically. Unsupervised means that the LLM algorithm of the future would not need explicit human-labeled data; instead, it would autonomously identify and reduce biases by analyzing patterns within the training data. As much as it sounds like a promising area of research, biases can be subtle and hard for algorithms to detect without human guidance, giving rise to the fear that new biases might get introduced in this process.

## Enhanced creativity

LLMs have exhibited creativity in text-based art forms like poetry and storytelling, and in conjunction with other AI models, they have produced visual art and music. They generate artistic content based on learned patterns from training data, not from emotional understanding or consciousness. Despite creating human-like artistic content, they don't comprehend art or emotions like humans. Some advancements are exploring the ability for LLMs to demonstrate human-like emotional behavior, enhancing the interaction between humans and computers. The future scope of LLMs in emotion inference is a subject of ongoing research and discussion.

