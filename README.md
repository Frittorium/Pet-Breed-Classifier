# Pet Breed Classification:

A PyTorch study comparing pretrained and custom convolutional networks for fine-grained classification of 37 cat and dog breeds, with a Tkinter desktop app for top-5 inference.

## Dataset

The project uses the Oxford-IIIT Pet dataset (`torchvision.datasets.OxfordIIITPet`), with 3,680 images from the `trainval` split across 37 breeds. Images vary in pose, scale, and lighting. Many breeds are visually similar, which makes the task fine-grained. The data was split 80/20 with a fixed seed (42), giving 2,944 training and 736 validation images.

## Objective

The goal is to identify a pet's breed from a single photograph and return the five most likely breeds with confidence scores. The project also asks how much a pretrained backbone helps compared with a CNN trained from scratch on a small dataset.

## Preprocessing and Augmentation

All inputs are normalized with ImageNet mean and standard deviation.

| Pipeline | Training | Validation |
| --- | --- | --- |
| ResNet18 | RandomResizedCrop(224), horizontal flip | Resize(256), CenterCrop(224) |
| Custom CNN | Resize(224×224), horizontal flip, rotation ±10° | Resize(224×224) |

Separate dataset instances share the same split indices, so augmentation is applied only to training data and validation stays deterministic.

## Architectures

- **ResNet18 (ImageNet-pretrained):** the final layer is replaced with a 37-way linear head.
- **Custom CNN:** five Conv→BatchNorm→ReLU→MaxPool blocks (32→64→128→256→512 channels), then adaptive average pooling to 4×4. This feeds a 256-unit fully connected layer with 0.5 dropout and a 37-class output.

## Design Rationale

- **Pretrained backbone:** with under 3,000 training images, features learned on ImageNet are a strong starting point, and training is fast.
- **Two adaptation levels:** freezing the backbone isolates the value of pretrained features. Unfreezing only `layer4` adapts the highest-level features to breed detail while keeping the general early filters.
- **Differential learning rates:** 1e-4 for `layer4` and 1e-3 for the new head keeps the pretrained weights from being disrupted.
- **Deployed model:** the transfer-learning model was saved for the desktop app rather than the fine-tuned one. Its training and validation curves stay close together (validation accuracy remains at or above training accuracy throughout), which indicates stable generalization. The fine-tuned model reaches slightly higher validation accuracy, but its training accuracy climbs above validation accuracy, an early sign of overfitting.
- **Custom CNN regularization:** batch normalization, dropout, weight decay (1e-4), and a `ReduceLROnPlateau` scheduler were used to give the scratch model a fair chance.

## Experiments

| Experiment | Trainable parameters | Epochs |
| --- | --- | --- |
| Transfer learning (frozen ResNet18, new head) | 18,981 | 10 |
| Fine-tuning (`layer4` + head) | 8,412,709 | 10 |
| Custom CNN, short run | all | 10 |
| Custom CNN, extended run | all | 50 |

All runs used Adam and cross-entropy loss.

## Results

| Model | Final val. accuracy | Best val. accuracy | Training time |
| --- | --- | --- | --- |
| Transfer learning | 90.4% | 91.0% (epoch 7) | \~1.1 min |
| **Fine-tuning** | **91.3%** | **92.4% (epoch 8)** | \~1.1 min |
| Custom CNN (10 epochs) | 3.9% | 3.9% | \~1.7 min |
| Custom CNN (50 epochs) | 2.0% | 2.0% | \~8.9 min |

## Key Findings

- Pretrained features are decisive. Even a linear head on a frozen backbone exceeds 90% validation accuracy in about a minute of training.
- Fine-tuning `layer4` gives a modest gain in validation accuracy (about one point). However, its training accuracy (92.7%) ends above its validation accuracy (91.3%), and the two diverge over the run, suggesting the added capacity begins to overfit. The transfer-learning model shows no such divergence: its validation accuracy (90.4%) stays above its training accuracy (85.0%) at the final epoch, partly because training images are heavily augmented. It was therefore chosen as the saved model for its more stable generalization, at a cost of roughly one point of accuracy.
- The custom CNN failed to learn. Its loss stayed near 3.61 and accuracy stayed near chance (about 2.7%), and the 50-epoch run did not improve on the 10-epoch run. The scratch model, with this dataset size and setup, could not learn usable features. Likely next steps are stronger augmentation, learning-rate tuning, or a shallower design.
