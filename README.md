# NLP Sentiment Analysis of IMDb Movie Reviews

### Project Overview

This project performs a comprehensive sentiment analysis on the IMDb Movie Reviews dataset. The core objective is to build, compare, and analyze the performance of two distinct text classification models: a classical machine learning approach and a modern deep learning architecture. This project showcases a complete and iterative data science workflow, moving from data preprocessing and baseline modeling to advanced model implementation, critical analysis, and drawing evidence-based conclusions.


### Dataset

The analysis was conducted on the "IMDb Dataset of 50K Movie Reviews," a large and well-balanced dataset ideal for binary sentiment classification. It contains 50,000 reviews, evenly split with 25,000 labeled as 'positive' and 25,000 as 'negative'.

[Link to the dataset on Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)


### Project Workflow

1.  **Data Preprocessing:** The raw text data was meticulously cleaned and prepared for modeling. This involved removing HTML tags, converting text to lowercase, stripping punctuation, and filtering out common English stop words to reduce noise.
2.  **Baseline Model:** A strong performance baseline was established using a classical NLP approach: text vectorization with **TF-IDF** (Term Frequency-Inverse Document Frequency) followed by a **Logistic Regression** classifier.
3.  **Advanced Model:** A deep learning model was implemented using a **Long Short-Term Memory (LSTM)** network. This architecture was chosen for its ability to capture sequential context and long-range dependencies within the text, which a bag-of-words model like TF-IDF ignores.
4.  **Initial Comparison & Analysis:** The initial results were surprising: the simpler baseline model outperformed the more complex LSTM. An analysis of the LSTM's learning curves suggested that the model was under-trained.
5.  **Iteration & Final Diagnosis:** To test the under-training hypothesis, the LSTM was retrained for more epochs, incorporating an **`EarlyStopping`** callback as a best practice. This experiment revealed that the LSTM began to overfit almost immediately after its first epoch and, even at its peak performance, could not surpass the baseline.


### Results and Key Findings

This project's most significant finding is a classic lesson in machine learning: **greater model complexity does not always yield better results.**


**Logistic Regression (TF-IDF)** ->   **89.14%** -> A highly effective and efficient baseline. Its superior performance suggests that the presence of specific keywords (captured by TF-IDF) is the strongest signal for sentiment in this dataset. 
**LSTM (Tuned w/ Early Stopping)** ->   **87.77%** -> While capable of understanding context, the model began to overfit quickly. Even at its optimal state, it was less effective than the simpler, more robust baseline. 

### Conclusion

After a thorough comparative analysis, the final recommendation is to use the **TF-IDF with Logistic Regression model** for this task. It is not only more accurate but also vastly more computationally efficient and easier to interpret.

This project demonstrates that while deep learning models are incredibly powerful, they are not always the superior solution. Establishing a strong baseline is a critical step in any machine learning workflow, as it provides a clear benchmark that more complex models must convincingly surpass to justify their implementation costs. The final conclusion is that for this specific problem, a robust statistical approach proved more effective than a nuanced sequential one.


### Technologies Used
- Python
- Pandas
- NLTK
- Scikit-learn
- TensorFlow (Keras)
- Matplotlib