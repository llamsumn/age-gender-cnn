# Lecturer Feedback Summary

**Module**: CM52053 - Artificial Intelligence and Machine Learning (Coursework 2)  
**Mark**: 82 / 100

## Score Breakdown

| Category | Criterion | Score |
|----------|-----------|-------|
| **Data Pre-processing** | Sample display & data augmentation | 3/4 |
| **Model A** | Model construction | 11/12 |
| | Effective training | 5/6 |
| | Model explanation | 7/8 |
| | Age performance (MAE: 6.63) | 8/9 |
| | Gender performance (Accuracy: 88.4%) | 7/9 |
| **Model B** | Model construction | 10/12 |
| | Effective training | 3/6 |
| | Model explanation | 7/8 |
| | Age performance (MAE: 6.76) | 7/9 |
| | Gender performance (Accuracy: 86.3%) | 7/9 |
| **Summary** | Comparison & discussion | 7/8 |

## Positive Feedback

- The document is well-structured and written in a very professional manner.
- Clear explanation of models, reasoning behind design choices, performance discussion, and possible improvements.
- The code demonstrates the author's capability in building and training CNN models for a real-life problem.
- Data augmentation (horizontal flipping) was applied.
- The multi-output model with common feature extraction was a good choice for the two tasks.
- Loss weights were adjusted to prevent training from biasing toward the age loss — the lecturer was glad to see this.
- Dropout and early stopping were used to prevent overfitting.
- Batch normalization and learning rate scheduling were employed to stabilise training.
- Model A is a reasonable size.
- The pretrained EfficientNetB0 was fine-tuned on higher conv layers with added classification layers.
- Strong, well-rounded summary with clear model comparison and insightful discussion of practical relevance.
- The two-step approach for fine-tuning was correctly used.

## Areas for Improvement

1. **Data Augmentation**: Too strong augmentation on a well-aligned face dataset may degrade performance. Consider experimenting with more subtle transformations beyond horizontal flipping.

2. **Model B — Global Average Pooling**: GAP reduces each feature map to a single number, which is an extreme dimension reduction that may lead to information loss. The lecturer suggests replacing it with direct flattening to see if performance improves.

3. **Model B — Overfitting**: Overfitting was observed for both tasks on Model B (and for the age task on Model A). This is a key area where marks were lost, particularly on Model B's "Effective training" criterion (3/6).

4. **Model B — Fine-tuning Strategy**: During the second phase, the whole convnet base was unfrozen. This is not necessary and is prone to overfitting since too many parameters need training for such a small dataset. Only the higher layers should be unfrozen for fine-tuning.

5. **Model B — Underperformance**: It is surprising that the pre-trained model performed worse than the custom model. The lecturer notes this means the potential of the pre-trained model has not been fully discovered — likely due to the overfitting and fine-tuning issues above.
