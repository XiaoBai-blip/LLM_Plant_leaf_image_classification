# LLM_Plant_leaf_image_classification
## Problem Statement
Plant diseases significantly impact global agriculture, reducing crop yields and threatening food security, yet traditional detection methods are slow, labor-intensive, and prone to error. To address this, we developed a hybrid pipeline that integrates the Segment Anything Model (SAM) with Vision Transformers (ViT) and ResNet architectures for accurate and robust plant disease detection. SAM was used to isolate disease-relevant regions in images, improving data quality and reducing noise, while ViT and ResNet were fine-tuned using both original and segmented datasets from the PlantVillage dataset. Our approach achieved a top-1 test accuracy of 99.95\% with ViT and 99.81\% with ResNet on the original dataset, and 99.55\% and 99.40\%, respectively, on SAM-segmented data. Robustness tests further demonstrated ViT’s superior generalization under noisy conditions. These results underscore the potential of combining segmentation with advanced vision models to improve the accuracy and scalability of plant disease detection, providing a powerful tool for sustainable agricultural practices.
## Proposed Approach
We integrated Vision Transformers (ViT) and ResNet with the Segment Anything Model (SAM) to address plant disease detection challenges. The choice of ViT and ResNet stems from their complementary strengths: ViT's global attention mechanism excels in capturing high-level features, while ResNet's hierarchical structure effectively captures local patterns. SAM enhances the dataset by segmenting images to focus on disease-relevant regions, improving data diversity and robustness.

Our methodology involves training and fine-tuning both ViT and ResNet on the PlantVillage dataset, consisting of 20,638 images across 15 disease categories. The training process is augmented using techniques such as gradient blur and rotation to improve model generalization. We evaluate models on original and SAM-segmented datasets, exploring their performance under noise and testing their robustness.

Key experimental steps:
1. **Baseline Zero-Shot Testing**: Conducted with pre-trained ViT and ResNet models on the original dataset.
2. **Model Training on Original Dataset**: Fine-tuning the models with standard training techniques to establish benchmark accuracies.
3. **Integration of SAM-Segmented Data**: Incorporating SAM-segmented images into training and testing pipelines to analyze performance improvements.
4. **Comparison of Models**: Analyzing ViT and ResNet performance metrics, including accuracy, robustness under noise, and inference times.

## Contributions

This study makes the following contributions:
1. **Model Development**: We integrate SAM with two advanced vision models (ViT and ResNet) to improve plant disease detection accuracy and robustness.
2. **Performance Comparison of Vision Models**: By comparing ViT and ResNet on both original and segmented datasets, we highlight their respective strengths and limitations, offering insights into their application in agricultural diagnostics.
3. **Enhanced Data Augmentation Techniques**: Incorporating gradient blur and rotation, we achieve better generalization and robustness, particularly in noisy environments.
4. **Robustness and Efficiency**: Our models demonstrate high robustness, with ViT achieving top-1 accuracy of 99.88% and ResNet achieving a comparable performance of 99.40%. The results underscore the effectiveness of SAM in augmenting plant disease datasets.

This research provides a foundation for scalable and efficient plant disease detection systems, contributing to the broader goal of sustainable agricultural practices.

## Result

The results presented in this section highlight the performance of our approach for plant disease classification using the Segmentation Anything Model (SAM) for segmentation and two visual language models: Vision Transformer (ViT) and ResNet50. The experiments were conducted on the original and SAM-processed datasets, and the outcomes are evaluated using metrics such as accuracy, F1-Score, and robustness tests. Key findings are summarized using tables, plots, and visualizations.

### Evaluation Metrics

We evaluated the performance of our models using the following metrics:

- **Accuracy**: The proportion of correctly classified images to the total number of images.
- **F1-Score (Macro)**: The harmonic mean of precision and recall, averaged across all classes.
- **Precision**: The proportion of correctly identified positive cases out of the total predicted positive cases.
- **Recall**: The proportion of correctly identified positive cases out of the total actual positive cases.
- **ROC-AUC (One-vs-Rest)**: The area under the receiver operating characteristic curve for each class.
- **Robustness Test Accuracy**: Accuracy under artificially added noise to evaluate the model's resilience to input perturbations.

### Main Results

The initial evaluation of the models using a zero-shot approach demonstrated their limitations in this specific domain. The Vision Transformer (ViT), pre-trained on ImageNet, achieved a test accuracy of 0.73%, while ResNet50, also pre-trained on ImageNet, performed slightly better with a test accuracy of 1.12%. These results are consistent with expectations since neither model has been pre-trained for plant disease classification. ViT's performance was constrained by its reliance on high-level feature representations, which did not translate well to the unique visual patterns in plant disease images without fine-tuning. In contrast, ResNet50 showed slightly better zero-shot performance due to its ability to extract low-level features like edges and textures, which occasionally aligned with certain disease-related patterns. However, both models struggled to generalize effectively in this domain without additional training, underscoring the importance of fine-tuning on a labeled dataset tailored to plant disease classification. This evaluation highlights the necessity of adapting pre-trained models to domain-specific tasks to achieve meaningful results.

#### Vision Transformer (ViT)
**Training Metrics on the Original Dataset:**

| Epoch | Training Accuracy | Loss   | Speed (it/s) | Notes                  |
|-------|--------------------|--------|--------------|------------------------|
| 1/4   | 93.71%            | 0.5581 | 3.28         | Initial convergence.   |
| 2/4   | 99.47%            | 0.0826 | 3.31         | Rapid improvement.     |
| 3/4   | 99.88%            | 0.0356 | 3.33         | Close to convergence.  |
| 4/4   | 99.99%            | 0.0214 | 3.32         | Achieved near-perfect. |

**Performance on Original Dataset:**

| Metric                | Training Accuracy | Test Accuracy | Average Inference Time (ms/image) | Robustness Test Accuracy (10% noise) |
|-----------------------|--------------------|---------------|------------------------------------|---------------------------------------|
| Epoch 4/4            | 99.99%            | 99.95%        | 0.16                              | 70.01%                                |

**Performance on SAM-Processed Dataset:**


| Metric                | Training Accuracy | Test Accuracy | Average Inference Time (ms/image) | Robustness Test Accuracy (10% noise) |
|-----------------------|--------------------|---------------|------------------------------------|---------------------------------------|
| Epoch 5/5            | 99.88%            | 99.55%        | 0.16                              | 46.58%                                |

#### ResNet50

**Performance on Original Dataset:**

| Metric                | Training Accuracy | Test Accuracy | Average Inference Time (ms/image) | Robustness Test Accuracy (10% noise) |
|-----------------------|--------------------|---------------|------------------------------------|---------------------------------------|
| Epoch 7/7            | 99.95%            | 99.81%        | 0.23                              | 49.10%                                |

**Performance on SAM-Processed Dataset:**

| Metric                | Training Accuracy | Test Accuracy | Average Inference Time (ms/image) | Robustness Test Accuracy (10% noise) |
|-----------------------|--------------------|---------------|------------------------------------|---------------------------------------|
| Epoch 8/8            | 99.68%            | 99.40%        | 0.17                              | 23.40%                                |

#### Confusion Matrix

The confusion matrix for ViT on the SAM dataset highlights the class-wise performance:

- Diagonal values represent correct classifications, with off-diagonal values indicating misclassifications.
- Most misclassifications occurred between similar disease classes, such as Tomato diseases.

The confusion matrix for the ViT model trained on the SAM dataset provides a comprehensive view of its class-wise performance. Each diagonal element represents the count of correctly classified images for a specific class, while the off-diagonal elements indicate the number of misclassifications. The majority of predictions are concentrated along the diagonal, reflecting the high accuracy of the model. However, certain classes, such as Tomato Early Blight and Tomato Late Blight, exhibit misclassifications due to their overlapping visual symptoms. Additionally, rare classes like Pepper Bell Bacterial Spot show slightly higher misclassification rates, which could be attributed to the limited number of training samples available for these classes. The confusion matrix emphasizes the model's strength in distinguishing most classes while also highlighting areas where additional data or refined feature representations could reduce misclassification rates.

####  Evaluation Metrics

The bar chart below illustrates the models precision, recall, F1-Score, accuracy, and other metrics:

- All metrics are above 0.99, showcasing the high performance of the model across all evaluation criteria.

The evaluation metrics chart underscores the remarkable performance of the Vision Transformer model across all key evaluation criteria. Metrics such as precision, recall, F1-Score, and accuracy consistently exceed **0.99**, showcasing the model's ability to accurately classify plant disease images. The precision metric highlights the model's strength in minimizing false positives, while the recall metric reflects its effectiveness in identifying true positive cases. The high F1-Score indicates a balanced trade-off between precision and recall, ensuring robust classification performance. Moreover, the near-perfect Top-3 and Top-5 accuracies demonstrate the model's capability to rank the correct class among the top predictions, even in cases where the top prediction is incorrect. These results reaffirm the model's reliability and adaptability for practical applications in plant disease diagnostics.

####  ROC Curves

The ROC curves for the ViT model demonstrate class-wise discrimination:


- AUC values for all classes exceeded 0.99, reflecting excellent model discrimination.

The ROC curve analysis provides additional insights into the model's performance. For each class, the curve plots the trade-off between the True Positive Rate (TPR) and False Positive Rate (FPR), with the area under the curve (AUC) serving as a summary statistic. The AUC values for all classes exceed **0.99**, indicating exceptional class-wise discrimination capabilities. The ROC curves are closely aligned with the top-left corner of the plot, reflecting the model's strong ability to differentiate between positive and negative cases across all classes. The inclusion of a random guess baseline further emphasizes the model's superiority, as all curves significantly outperform the diagonal line. These findings highlight the model's effectiveness in multi-class classification tasks, making it a reliable tool for plant disease detection.

### Comparison of ViT and ResNet50

The comparison between Vision Transformer (ViT) and ResNet50 reveals key differences in their performance across original and SAM-processed datasets. ViT consistently outperformed ResNet50 in terms of test accuracy and robustness. For the original dataset, ViT achieved a test accuracy of **99.95%**, compared to ResNet50's **99.81%**. On the SAM-processed dataset, ViT maintained a higher test accuracy of **99.55%**, while ResNet50 achieved **99.40%**. In terms of robustness, ViT demonstrated superior performance under noise conditions, with robustness test accuracies of **70.01%** and **46.58%** on the original and SAM datasets, respectively, compared to ResNet50's **49.10%** and **23.40%**. These results highlight ViT's ability to generalize better across different data preprocessing methods. However, ResNet50 exhibited slightly faster inference times, making it more suitable for applications requiring real-time predictions. Overall, ViT emerged as the preferred model for scenarios prioritizing accuracy and robustness, while ResNet50's computational efficiency makes it a viable option for resource-constrained settings.

**Summary of Results:**

| Model      | Test Accuracy (Original) | Test Accuracy (SAM) | Robustness Test Accuracy (Original) | Robustness Test Accuracy (SAM) |
|------------|---------------------------|----------------------|--------------------------------------|---------------------------------|
| ViT        | 99.95%                   | 99.55%              | 70.01%                              | 46.58%                         |
| ResNet50   | 99.81%                   | 99.40%              | 49.10%                              | 23.40%                         |

- **Accuracy**: ViT outperformed ResNet50 in both original and SAM-processed datasets.
- **Robustness**: ViT demonstrated higher robustness than ResNet50, particularly on the original dataset.
- **Inference Time**: ResNet50 had a slightly higher inference time per image compared to ViT.

ViT emerged as the superior model in terms of accuracy and robustness across both datasets. While ResNet50's architectural simplicity made it computationally efficient, its performance was comparatively lower, particularly on SAM-processed data. The results highlight the trade-offs between computational efficiency and accuracy, suggesting that ViT is better suited for applications where high accuracy is prioritized, whereas ResNet50 may be preferable for real-time use cases.


## For full conducting process refer Report file
