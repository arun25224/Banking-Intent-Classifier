# Banking Intent Classifier

An end-to-end natural language processing system designed to classify customer banking queries across 77 distinct intent categories. In financial environments, traditional keyword matching frequently fails on nuanced language, causing misclassification of high-risk requests such as compromised cards. To resolve this, the system transitions from a baseline Multi-Layer Perceptron (MLP) to a bidirectional RoBERTa transformer fine-tuned with Low-Rank Adaptation (LoRA), achieving a 93.80% Macro F1 score on a balanced evaluation suite.


<img width="235" height="69" alt="image" src="https://github.com/user-attachments/assets/81994c5d-de51-47f5-b43c-1f10dcca5cce" />

*Dataset split comprising 10,003 training examples and 3,080 test examples across 77 banking intent classes.*

## Technical Architecture

* **Data Ingestion and PII Sanitization:** Raw user queries pass through an automated regular-expression sanitization pipeline prior to model exposure. The system detects and redacts sensitive Personally Identifiable Information (PII), such as 16-digit payment card numbers, email addresses, and phone numbers.
* **Baseline Classifier (TF-IDF + MLP):** To measure performance gains against lower-complexity baselines, input text was vectorized via Term Frequency-Inverse Document Frequency (TF-IDF) and evaluated on a multi-layer perceptron. While computationally light, this architecture disregards sequence order and semantic context.
* **Transformer Fine-Tuning (RoBERTa):** Upgraded to a pre-trained RoBERTa model to capture bidirectional contextual dependencies across user requests.
* **Parameter-Efficient Adaptation (LoRA):** Implemented Low-Rank Adaptation with rank r = 64 across attention modules (Query, Key, Value, and Dense layers). This froze base model parameters and isolated gradient updates to low-rank matrices, allowing training to run on standard GPU hardware without enterprise cluster overhead.
* **Optimization and Regularization:** Utilized a linear warmup learning rate scheduler to protect pre-trained weights from early gradient instability, combined with gradient accumulation and dynamic memory cleanup to avoid Out-Of-Memory (OOM) exceptions.

<img width="1190" height="547" alt="image" src="https://github.com/user-attachments/assets/bb277a11-66af-40fd-b251-3dd5198eb9b7" />
*Distribution of the ten most frequent customer intents, showing high volume in card payment fees, direct debit disputes, and deposit balance updates.*

## Results and Evaluation

The model was evaluated using Macro F1 alongside classification accuracy across all 77 intent categories, ensuring that performance metrics accurately reflect classification success on rare classes rather than skewing toward majority administrative intents.

| Model Architecture | Compute Overhead | Macro F1 Score | Accuracy |
| :--- | :--- | :--- | :--- |
| MLP Baseline (TF-IDF) | Low | 88.07% | 88.02% |
| RoBERTa (LoRA) | Moderate | 93.80% | 93.80% |

<img width="381" height="84" alt="image" src="https://github.com/user-attachments/assets/bb35b904-4d6e-4983-a367-f319a27b300b" />

The LoRA-adapted RoBERTa model yielded an absolute gain of over 5.7% in Macro F1 over the baseline. Because the Macro F1 score matched overall accuracy at 93.80%, the system demonstrates balanced generalization across tail-distribution intents.

<img width="318" height="375" alt="image" src="https://github.com/user-attachments/assets/cdb6ea18-45f8-495a-92a8-5f7295583bc0" />
*Ten-epoch training run illustrating steady reduction in validation loss down to 0.257, reaching a final Macro F1 score of 0.9380.*

## Production Deployment Roadmap

* **Confidence Routing:** Implement a decision threshold where any inference with a predicted probability below 0.50 is automatically escalated to human agents to prevent automated misrouting.
* **Active Learning Feedback:** Flagged, human-resolved low-confidence queries are logged into an asynchronous review database and added to periodic LoRA retraining cycles.
* **Inference Quantization:** Apply 8-bit or 4-bit quantization post-training to reduce memory footprint and maintain sub-50ms latency for production live-chat systems.

## Project Dependencies

* Python 3.x
* PyTorch
* Hugging Face Transformers
* Hugging Face PEFT (Parameter-Efficient Fine-Tuning)
* scikit-learn
* pandas
* numpy
* matplotlib
