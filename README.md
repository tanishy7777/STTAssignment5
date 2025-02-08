GROUP 28

Laksh Jain - 23110185

Tanish Yelgoe - 23110

TASK-1

For data augmentation, we randomly sample 3 augmentations from a set of 10 augmentations - rotation by 30 degrees, scale, saturation, blur, random-noise, colour jitter, pad square, pad, hflip and pixelization, and then apply the 3 augmentations on the image sequentially. We do this process twice for every image.

IMAGE STATISTICS

Initially we have 112 images in train set and 28 images in test set. The number of images of cats and dogs is EQUAL in both train and test set.

For every image in train set, we apply augmentation function (on applying function once, the image is augmented thrice) 2 times. Hence, the number of augmented images is 112 * 2 = 224. Hence, total number of images in train set is 112 + 224 = 336.

TASK-2

We load the resnet model from huggingface. Below is the architecture of the resnet model.

![image-2.png](attachment:image-2.png)

We load the pre-trained weights of the model, and define 2 models: model_non_aug and model_aug with the same weights.

```python
# Get the configuration from the Hugging Face hub.
config = ResNetConfig.from_pretrained("microsoft/resnet-50", num_labels=num_classes)

# Initialize a model with new (random) weights.
model_init = ResNetForImageClassification(config)
# Save the initial state dict.
initial_state_dict = copy.deepcopy(model_init.state_dict())

# Create two separate model instances and load the same initial weights.
model_non_aug = ResNetForImageClassification(config)
model_non_aug.load_state_dict(copy.deepcopy(initial_state_dict))

model_aug = ResNetForImageClassification(config)
model_aug.load_state_dict(copy.deepcopy(initial_state_dict))

# Move models to device.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_non_aug.to(device)
model_aug.to(device)

```
The below is the information about the training parameters of the models.

