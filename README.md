# Next Word Predictor using LSTM

## Project Overview
The **Next Word Predictor** is a Natural Language Processing (NLP) project built using Long Short-Term Memory (LSTM) networks. This project demonstrates the capability of deep learning to predict the next word in a given text sequence, leveraging sequential data and contextual understanding. It showcases the power of LSTM for language modeling and text generation tasks.

---

## Key Features
1. **Language Modeling**:
   - Uses LSTM networks to understand and generate human-like text sequences.
   - Trained on large text datasets to learn word patterns and semantic relationships.

2. **Next Word Prediction**:
   - Given an input sequence, the model predicts the most probable next word.
   - Iterative predictions allow generation of entire sentences.

3. **Deep Learning Framework**:
   - Employs Keras for building and training the LSTM model.
   - Utilizes tokenization and padding to preprocess textual data for sequential learning.

4. **Training Process**:
   - The model was trained over 100 epochs, achieving significant improvement in accuracy (up to 99%).
   - Loss reduction and performance metrics highlight the learning progress.

5. **Interactive Text Generation**:
   - Accepts user input to generate predictions iteratively, creating coherent text outputs.

---

## Implementation Details

### 1. **Data Preprocessing**:
   - Tokenization: Converts text into sequences of numeric tokens.
   - Padding: Ensures uniform input length for training the model.
   - One-hot encoding: Converts target words into categorical data for prediction.

### 2. **Model Architecture**:
   - **Embedding Layer**: Creates word embeddings for semantic representation.
   - **LSTM Layer**: Learns sequential patterns in text data.
   - **Dense Layer**: Outputs probabilities for the next word using a softmax activation function.

### 3. **Training Details**:
   - Optimizer: Adam
   - Loss Function: Categorical Crossentropy
   - Training Dataset: Custom text corpus with sequential data.

### 4. **Prediction Logic**:
   - The model takes input sequences and predicts the next word iteratively.
   - Context-awareness allows for grammatically and semantically appropriate text generation.

---

## Results
- **Accuracy**: Achieved up to 99% accuracy on the training dataset after 100 epochs.
- **Text Generation**: Produces coherent and contextually relevant sentences based on input sequences.
- **Performance Improvements**: Demonstrated loss reduction from initial epochs to convergence.

---

## Tools and Technologies Used
- **Programming Language**: Python
- **Frameworks and Libraries**: TensorFlow, Keras, NumPy, Pandas, Matplotlib
- **Visualization**: Training accuracy and loss were plotted to monitor performance.

---

## Applications
- **Content Generation**: Assists in generating text for articles, stories, and creative writing.
- **Chatbots**: Enhances conversational AI systems by predicting user intents and responses.
- **Language Understanding**: Provides a foundation for tasks like autocomplete and predictive text.

---

## Future Enhancements
- Train on larger and diverse datasets for improved generalization.
- Extend the model to handle multilingual text.
- Incorporate attention mechanisms for enhanced context understanding.

---

## How to Run
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`.
3. Run the script: `python next_word_predictor.py`.
4. Enter a text sequence to generate predictions.

