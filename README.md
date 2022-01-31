#LIHT: Label Informed Hierarchical Transformers for Sequential Sentence Classification in Scientific Abstracts

This is the repository containing the code to our LIHT paper. Use the train.py script to run the training / evaluation steps.

---
### Paper abstract
Segmenting scientific abstracts into several discourse categories like background, objective, method, result, and conclusion is helpful in many downstream tasks like search, recommendation and summarization. Previous approaches for sequential sentence classification consider the content of the abstract only to obtain the representation of each sentence, thereby ignoring the semantic information offered by the discourse labels. In this work, we propose LIHT, Label Informed Hierarchical Transformers for sequential sentence classification which explicitly exploits the semantic information in the labels hierarchically to learn label-aware sentence representation. This hierarchical modeling helps us to capture not only the fine-grained interactions between the words and discourse labels at the word level, but also helps to identify potential dependencies that may exist in the label sequence by providing each sentence with a refined label-aware representation in each layer. Experimental results on three publicly available datasets demonstrate the effectiveness of our proposed method.
