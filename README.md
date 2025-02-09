# cxd_pneu_classifier
Classification of Chest X-Ray Images (Pneumonia) using Pytorch

Dataset used for training was obtained from kaggle: [link](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

#### Information about project
- Two classification classes: Pnenumonia and Healthy (Pneumonia class can be extended to include bacterial and virus pneumonia)
- Average accuracy: 84%
- Training loop utilizes tensorboard for better analysis

#### Built with
- pytorch, torchvision
- matplotlib
- tensorboard
- sklearn (for metrics)

#### ResNet newtork

A pre-trained ResNet18 model was fine-tuned on a chest X-ray dataset, resulting in a much higher accuracy compared to previous architecture. The model achieves 96.31% accuracy on the test dataset.
