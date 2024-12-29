# Sentiment Analysis of Video Game Reviews

## Project Overview
This project focuses on sentiment analysis of video game reviews, leveraging advanced natural language processing techniques to classify reviews as positive or negative. Despite the dataset's high imbalance (80% positive reviews), the project achieved significant accuracy improvements through data augmentation and model optimization.

---

## Dataset
- **Source:** Steam Games Reviews
- **Nature:** Highly imbalanced dataset (80% positive reviews)

---

## Methodology
### Data Augmentation
- Utilized **NLPaug** for contextual data augmentation.
- Augmented minority class samples using **RoBERTa** to enhance class balance.

### Models Implemented
1. **LSTM (4 layers)**
   - Accuracy: **86%**
2. **Bi-LSTM (3 layers)**
   - Accuracy: **85%**
3. **BERT Sequence Classifier**
   - Trained classifier layers only: **82% accuracy**
4. **BERT Sequence Classifier with LoRA**
   - Trained classifier layers with additional layers using **LoRA (Low-Rank Adaptation)** and **4-bit quantization**: **92% accuracy**

---

## Highlights
- **Data Augmentation:** Improved class balance with contextual augmentation using **RoBERTa**.
- **Progressive Model Development:**
  - Transitioned from basic LSTM models to transformer-based architectures.
  - Implemented **LoRA** for parameter-efficient fine-tuning.
  - Optimized performance and resource utilization using **4-bit quantization**.
- Achieved a significant accuracy boost (92%) with the advanced BERT-based approach.

---

## Dependencies
- **Python 3.8+**
- **PyTorch 1.12+**
- **Hugging Face Transformers**
- **NLPaug**
- **RoBERTa Pretrained Model**

Install dependencies using:
```bash
pip install torch transformers nlpaug
```

---

[//]: # (## How to Run)

[//]: # (1. Clone the repository:)

[//]: # (   ```bash)

[//]: # (   git clone <repository_url>)

[//]: # (   cd <repository_name>)

[//]: # (   ```)

[//]: # (2. Install dependencies &#40;see above&#41;.)

[//]: # (3. Prepare the dataset:)

[//]: # (   - Place the Steam reviews dataset in the `data/` directory.)

[//]: # (   - Ensure the file structure matches the preprocessing script requirements.)

[//]: # (4. Run the training script:)

[//]: # (   ```bash)

[//]: # (   python train.py --model <model_name> --augment <True/False>)

[//]: # (   ```)

[//]: # (   Replace `<model_name>` with `lstm`, `bi-lstm`, or `bert`.)

[//]: # ()
[//]: # (---)

## Results
| Model                         | Accuracy |
|-------------------------------|----------|
| LSTM (4 layers)               | 86%      |
| Bi-LSTM (3 layers)            | 85%      |
| BERT Sequence Classifier      | 82%      |
| BERT + LoRA + 4-bit Quantization | 92%      |

---

## Future Work
- Explore other transformer architectures like **DeBERTa** or **DistilBERT**.
- Fine-tune models on a broader set of game reviews from other platforms.
- Implement more robust augmentation strategies.

---

[//]: # (## License)

[//]: # (This project is licensed under the MIT License. See the [LICENSE]&#40;LICENSE&#41; file for details.)

[//]: # ()
[//]: # (---)

## Acknowledgments
- Hugging Face for the Transformers library.
- Steam community for the dataset.
- NLPaug library for augmentation techniques.
