# Food classification
This is the final project for the Supervised Learning course at Unimib.
This is a group project developed along with my uni husband @MirkoMorello. 

## Summary
### Objective
Implement a CNN and classify food images (251 classes).
**Constraint**: The model cannot exceed 1M parameters. The network has to be completely custom.

### Approach
The data has several realistic problems, such as variable size of the image, unbalanced and under represented classes and mislabeled data. Since we had to compare performances with other collegues we did not, although in a realistic case we would consider it, to at least clean the mislabeled data.


We came up with TinyNet, our custom CNN shy of 1M parameters with stability in mind, hence the use of GELU as activation function, Max pooling, batch normalization and dropout (for the linear layer only). 
We pretrained the network with a Self-supervised approach. We trained it on a pretext task where the network (mirrored to make a unet-like design) had to fill blacked out regions of the training set images.
The weights were then transferred for the actual supervised task.
Although the pretrained has not significantly increase performances it has shown a faster and more stable convergence.

To further increase the performance of the network we tuned hyperparameters, specifically the size of the kernels of each convolutional layer and the size of the linear layers, using the model selection framework Optuna.

---
See the report for the results and further details