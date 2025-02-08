GROUP 28

Laksh Jain - 23110185

Tanish Yelgoe - 23110328

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

<div style="overflow-y: scroll; height: 150px; border: 1px solid #ddd; padding: 5px;">
  
Layer: resnet.embedder.embedder.convolution.weight | Size: torch.Size([64, 3, 7, 7]) | Requires Grad: True
Layer: resnet.embedder.embedder.normalization.weight | Size: torch.Size([64]) | Requires Grad: True
Layer: resnet.embedder.embedder.normalization.bias | Size: torch.Size([64]) | Requires Grad: True
Layer: resnet.encoder.stages.0.layers.0.shortcut.convolution.weight | Size: torch.Size([256, 64, 1, 1]) | Requires Grad: True
Layer: resnet.encoder.stages.0.layers.0.shortcut.normalization.weight | Size: torch.Size([256]) | Requires Grad: True
Layer: resnet.encoder.stages.0.layers.0.shortcut.normalization.bias | Size: torch.Size([256]) | Requires Grad: True
Layer: resnet.encoder.stages.0.layers.0.layer.0.convolution.weight | Size: torch.Size([64, 64, 1, 1]) | Requires Grad: True
Layer: resnet.encoder.stages.0.layers.0.layer.0.normalization.weight | Size: torch.Size([64]) | Requires Grad: True
Layer: resnet.encoder.stages.0.layers.0.layer.0.normalization.bias | Size: torch.Size([64]) | Requires Grad: True
Layer: resnet.encoder.stages.0.layers.0.layer.1.convolution.weight | Size: torch.Size([64, 64, 3, 3]) | Requires Grad: True
Layer: resnet.encoder.stages.0.layers.0.layer.1.normalization.weight | Size: torch.Size([64]) | Requires Grad: True
Layer: resnet.encoder.stages.0.layers.0.layer.1.normalization.bias | Size: torch.Size([64]) | Requires Grad: True
Layer: resnet.encoder.stages.0.layers.0.layer.2.convolution.weight | Size: torch.Size([256, 64, 1, 1]) | Requires Grad: True
Layer: resnet.encoder.stages.0.layers.0.layer.2.normalization.weight | Size: torch.Size([256]) | Requires Grad: True
Layer: resnet.encoder.stages.0.layers.0.layer.2.normalization.bias | Size: torch.Size([256]) | Requires Grad: True
Layer: resnet.encoder.stages.0.layers.1.layer.0.convolution.weight | Size: torch.Size([64, 256, 1, 1]) | Requires Grad: True
Layer: resnet.encoder.stages.0.layers.1.layer.0.normalization.weight | Size: torch.Size([64]) | Requires Grad: True
Layer: resnet.encoder.stages.0.layers.1.layer.0.normalization.bias | Size: torch.Size([64]) | Requires Grad: True
Layer: resnet.encoder.stages.0.layers.1.layer.1.convolution.weight | Size: torch.Size([64, 64, 3, 3]) | Requires Grad: True
Layer: resnet.encoder.stages.0.layers.1.layer.1.normalization.weight | Size: torch.Size([64]) | Requires Grad: True
Layer: resnet.encoder.stages.0.layers.1.layer.1.normalization.bias | Size: torch.Size([64]) | Requires Grad: True
Layer: resnet.encoder.stages.0.layers.1.layer.2.convolution.weight | Size: torch.Size([256, 64, 1, 1]) | Requires Grad: True
Layer: resnet.encoder.stages.0.layers.1.layer.2.normalization.weight | Size: torch.Size([256]) | Requires Grad: True
Layer: resnet.encoder.stages.0.layers.1.layer.2.normalization.bias | Size: torch.Size([256]) | Requires Grad: True
Layer: resnet.encoder.stages.0.layers.2.layer.0.convolution.weight | Size: torch.Size([64, 256, 1, 1]) | Requires Grad: True
Layer: resnet.encoder.stages.0.layers.2.layer.0.normalization.weight | Size: torch.Size([64]) | Requires Grad: True
Layer: resnet.encoder.stages.0.layers.2.layer.0.normalization.bias | Size: torch.Size([64]) | Requires Grad: True
Layer: resnet.encoder.stages.0.layers.2.layer.1.convolution.weight | Size: torch.Size([64, 64, 3, 3]) | Requires Grad: True
Layer: resnet.encoder.stages.0.layers.2.layer.1.normalization.weight | Size: torch.Size([64]) | Requires Grad: True
Layer: resnet.encoder.stages.0.layers.2.layer.1.normalization.bias | Size: torch.Size([64]) | Requires Grad: True
Layer: resnet.encoder.stages.0.layers.2.layer.2.convolution.weight | Size: torch.Size([256, 64, 1, 1]) | Requires Grad: True
Layer: resnet.encoder.stages.0.layers.2.layer.2.normalization.weight | Size: torch.Size([256]) | Requires Grad: True
Layer: resnet.encoder.stages.0.layers.2.layer.2.normalization.bias | Size: torch.Size([256]) | Requires Grad: True
Layer: resnet.encoder.stages.1.layers.0.shortcut.convolution.weight | Size: torch.Size([512, 256, 1, 1]) | Requires Grad: True
Layer: resnet.encoder.stages.1.layers.0.shortcut.normalization.weight | Size: torch.Size([512]) | Requires Grad: True
Layer: resnet.encoder.stages.1.layers.0.shortcut.normalization.bias | Size: torch.Size([512]) | Requires Grad: True
Layer: resnet.encoder.stages.1.layers.0.layer.0.convolution.weight | Size: torch.Size([128, 256, 1, 1]) | Requires Grad: True
Layer: resnet.encoder.stages.1.layers.0.layer.0.normalization.weight | Size: torch.Size([128]) | Requires Grad: True
Layer: resnet.encoder.stages.1.layers.0.layer.0.normalization.bias | Size: torch.Size([128]) | Requires Grad: True
Layer: resnet.encoder.stages.1.layers.0.layer.1.convolution.weight | Size: torch.Size([128, 128, 3, 3]) | Requires Grad: True
Layer: resnet.encoder.stages.1.layers.0.layer.1.normalization.weight | Size: torch.Size([128]) | Requires Grad: True
Layer: resnet.encoder.stages.1.layers.0.layer.1.normalization.bias | Size: torch.Size([128]) | Requires Grad: True
Layer: resnet.encoder.stages.1.layers.0.layer.2.convolution.weight | Size: torch.Size([512, 128, 1, 1]) | Requires Grad: True
Layer: resnet.encoder.stages.1.layers.0.layer.2.normalization.weight | Size: torch.Size([512]) | Requires Grad: True
Layer: resnet.encoder.stages.1.layers.0.layer.2.normalization.bias | Size: torch.Size([512]) | Requires Grad: True
Layer: resnet.encoder.stages.1.layers.1.layer.0.convolution.weight | Size: torch.Size([128, 512, 1, 1]) | Requires Grad: True
Layer: resnet.encoder.stages.1.layers.1.layer.0.normalization.weight | Size: torch.Size([128]) | Requires Grad: True
Layer: resnet.encoder.stages.1.layers.1.layer.0.normalization.bias | Size: torch.Size([128]) | Requires Grad: True
Layer: resnet.encoder.stages.1.layers.1.layer.1.convolution.weight | Size: torch.Size([128, 128, 3, 3]) | Requires Grad: True
Layer: resnet.encoder.stages.1.layers.1.layer.1.normalization.weight | Size: torch.Size([128]) | Requires Grad: True
Layer: resnet.encoder.stages.1.layers.1.layer.1.normalization.bias | Size: torch.Size([128]) | Requires Grad: True
Layer: resnet.encoder.stages.1.layers.1.layer.2.convolution.weight | Size: torch.Size([512, 128, 1, 1]) | Requires Grad: True
Layer: resnet.encoder.stages.1.layers.1.layer.2.normalization.weight | Size: torch.Size([512]) | Requires Grad: True
Layer: resnet.encoder.stages.1.layers.1.layer.2.normalization.bias | Size: torch.Size([512]) | Requires Grad: True
Layer: resnet.encoder.stages.1.layers.2.layer.0.convolution.weight | Size: torch.Size([128, 512, 1, 1]) | Requires Grad: True
Layer: resnet.encoder.stages.1.layers.2.layer.0.normalization.weight | Size: torch.Size([128]) | Requires Grad: True
Layer: resnet.encoder.stages.1.layers.2.layer.0.normalization.bias | Size: torch.Size([128]) | Requires Grad: True
Layer: resnet.encoder.stages.1.layers.2.layer.1.convolution.weight | Size: torch.Size([128, 128, 3, 3]) | Requires Grad: True
Layer: resnet.encoder.stages.1.layers.2.layer.1.normalization.weight | Size: torch.Size([128]) | Requires Grad: True
Layer: resnet.encoder.stages.1.layers.2.layer.1.normalization.bias | Size: torch.Size([128]) | Requires Grad: True
Layer: resnet.encoder.stages.1.layers.2.layer.2.convolution.weight | Size: torch.Size([512, 128, 1, 1]) | Requires Grad: True
Layer: resnet.encoder.stages.1.layers.2.layer.2.normalization.weight | Size: torch.Size([512]) | Requires Grad: True
Layer: resnet.encoder.stages.1.layers.2.layer.2.normalization.bias | Size: torch.Size([512]) | Requires Grad: True
Layer: resnet.encoder.stages.1.layers.3.layer.0.convolution.weight | Size: torch.Size([128, 512, 1, 1]) | Requires Grad: True
Layer: resnet.encoder.stages.1.layers.3.layer.0.normalization.weight | Size: torch.Size([128]) | Requires Grad: True
Layer: resnet.encoder.stages.1.layers.3.layer.0.normalization.bias | Size: torch.Size([128]) | Requires Grad: True
Layer: resnet.encoder.stages.1.layers.3.layer.1.convolution.weight | Size: torch.Size([128, 128, 3, 3]) | Requires Grad: True
Layer: resnet.encoder.stages.1.layers.3.layer.1.normalization.weight | Size: torch.Size([128]) | Requires Grad: True
Layer: resnet.encoder.stages.1.layers.3.layer.1.normalization.bias | Size: torch.Size([128]) | Requires Grad: True
Layer: resnet.encoder.stages.1.layers.3.layer.2.convolution.weight | Size: torch.Size([512, 128, 1, 1]) | Requires Grad: True
Layer: resnet.encoder.stages.1.layers.3.layer.2.normalization.weight | Size: torch.Size([512]) | Requires Grad: True
Layer: resnet.encoder.stages.1.layers.3.layer.2.normalization.bias | Size: torch.Size([512]) | Requires Grad: True
Layer: resnet.encoder.stages.2.layers.0.shortcut.convolution.weight | Size: torch.Size([1024, 512, 1, 1]) | Requires Grad: True
Layer: resnet.encoder.stages.2.layers.0.shortcut.normalization.weight | Size: torch.Size([1024]) | Requires Grad: True
Layer: resnet.encoder.stages.2.layers.0.shortcut.normalization.bias | Size: torch.Size([1024]) | Requires Grad: True
Layer: resnet.encoder.stages.2.layers.0.layer.0.convolution.weight | Size: torch.Size([256, 512, 1, 1]) | Requires Grad: True
Layer: resnet.encoder.stages.2.layers.0.layer.0.normalization.weight | Size: torch.Size([256]) | Requires Grad: True
Layer: resnet.encoder.stages.2.layers.0.layer.0.normalization.bias | Size: torch.Size([256]) | Requires Grad: True
Layer: resnet.encoder.stages.2.layers.0.layer.1.convolution.weight | Size: torch.Size([256, 256, 3, 3]) | Requires Grad: True
Layer: resnet.encoder.stages.2.layers.0.layer.1.normalization.weight | Size: torch.Size([256]) | Requires Grad: True
Layer: resnet.encoder.stages.2.layers.0.layer.1.normalization.bias | Size: torch.Size([256]) | Requires Grad: True
Layer: resnet.encoder.stages.2.layers.0.layer.2.convolution.weight | Size: torch.Size([1024, 256, 1, 1]) | Requires Grad: True
Layer: resnet.encoder.stages.2.layers.0.layer.2.normalization.weight | Size: torch.Size([1024]) | Requires Grad: True
Layer: resnet.encoder.stages.2.layers.0.layer.2.normalization.bias | Size: torch.Size([1024]) | Requires Grad: True
Layer: resnet.encoder.stages.2.layers.1.layer.0.convolution.weight | Size: torch.Size([256, 1024, 1, 1]) | Requires Grad: True
Layer: resnet.encoder.stages.2.layers.1.layer.0.normalization.weight | Size: torch.Size([256]) | Requires Grad: True
Layer: resnet.encoder.stages.2.layers.1.layer.0.normalization.bias | Size: torch.Size([256]) | Requires Grad: True
Layer: resnet.encoder.stages.2.layers.1.layer.1.convolution.weight | Size: torch.Size([256, 256, 3, 3]) | Requires Grad: True
Layer: resnet.encoder.stages.2.layers.1.layer.1.normalization.weight | Size: torch.Size([256]) | Requires Grad: True
Layer: resnet.encoder.stages.2.layers.1.layer.1.normalization.bias | Size: torch.Size([256]) | Requires Grad: True
Layer: resnet.encoder.stages.2.layers.1.layer.2.convolution.weight | Size: torch.Size([1024, 256, 1, 1]) | Requires Grad: True
Layer: resnet.encoder.stages.2.layers.1.layer.2.normalization.weight | Size: torch.Size([1024]) | Requires Grad: True
Layer: resnet.encoder.stages.2.layers.1.layer.2.normalization.bias | Size: torch.Size([1024]) | Requires Grad: True
Layer: resnet.encoder.stages.2.layers.2.layer.0.convolution.weight | Size: torch.Size([256, 1024, 1, 1]) | Requires Grad: True
Layer: resnet.encoder.stages.2.layers.2.layer.0.normalization.weight | Size: torch.Size([256]) | Requires Grad: True
Layer: resnet.encoder.stages.2.layers.2.layer.0.normalization.bias | Size: torch.Size([256]) | Requires Grad: True
Layer: resnet.encoder.stages.2.layers.2.layer.1.convolution.weight | Size: torch.Size([256, 256, 3, 3]) | Requires Grad: True
Layer: resnet.encoder.stages.2.layers.2.layer.1.normalization.weight | Size: torch.Size([256]) | Requires Grad: True
Layer: resnet.encoder.stages.2.layers.2.layer.1.normalization.bias | Size: torch.Size([256]) | Requires Grad: True
Layer: resnet.encoder.stages.2.layers.2.layer.2.convolution.weight | Size: torch.Size([1024, 256, 1, 1]) | Requires Grad: True
Layer: resnet.encoder.stages.2.layers.2.layer.2.normalization.weight | Size: torch.Size([1024]) | Requires Grad: True
Layer: resnet.encoder.stages.2.layers.2.layer.2.normalization.bias | Size: torch.Size([1024]) | Requires Grad: True
Layer: resnet.encoder.stages.2.layers.3.layer.0.convolution.weight | Size: torch.Size([256, 1024, 1, 1]) | Requires Grad: True
Layer: resnet.encoder.stages.2.layers.3.layer.0.normalization.weight | Size: torch.Size([256]) | Requires Grad: True
Layer: resnet.encoder.stages.2.layers.3.layer.0.normalization.bias | Size: torch.Size([256]) | Requires Grad: True
Layer: resnet.encoder.stages.2.layers.3.layer.1.convolution.weight | Size: torch.Size([256, 256, 3, 3]) | Requires Grad: True
Layer: resnet.encoder.stages.2.layers.3.layer.1.normalization.weight | Size: torch.Size([256]) | Requires Grad: True
Layer: resnet.encoder.stages.2.layers.3.layer.1.normalization.bias | Size: torch.Size([256]) | Requires Grad: True
Layer: resnet.encoder.stages.2.layers.3.layer.2.convolution.weight | Size: torch.Size([1024, 256, 1, 1]) | Requires Grad: True
Layer: resnet.encoder.stages.2.layers.3.layer.2.normalization.weight | Size: torch.Size([1024]) | Requires Grad: True
Layer: resnet.encoder.stages.2.layers.3.layer.2.normalization.bias | Size: torch.Size([1024]) | Requires Grad: True
Layer: resnet.encoder.stages.2.layers.4.layer.0.convolution.weight | Size: torch.Size([256, 1024, 1, 1]) | Requires Grad: True
Layer: resnet.encoder.stages.2.layers.4.layer.0.normalization.weight | Size: torch.Size([256]) | Requires Grad: True
Layer: resnet.encoder.stages.2.layers.4.layer.0.normalization.bias | Size: torch.Size([256]) | Requires Grad: True
Layer: resnet.encoder.stages.2.layers.4.layer.1.convolution.weight | Size: torch.Size([256, 256, 3, 3]) | Requires Grad: True
Layer: resnet.encoder.stages.2.layers.4.layer.1.normalization.weight | Size: torch.Size([256]) | Requires Grad: True
Layer: resnet.encoder.stages.2.layers.4.layer.1.normalization.bias | Size: torch.Size([256]) | Requires Grad: True
Layer: resnet.encoder.stages.2.layers.4.layer.2.convolution.weight | Size: torch.Size([1024, 256, 1, 1]) | Requires Grad: True
Layer: resnet.encoder.stages.2.layers.4.layer.2.normalization.weight | Size: torch.Size([1024]) | Requires Grad: True
Layer: resnet.encoder.stages.2.layers.4.layer.2.normalization.bias | Size: torch.Size([1024]) | Requires Grad: True
Layer: resnet.encoder.stages.2.layers.5.layer.0.convolution.weight | Size: torch.Size([256, 1024, 1, 1]) | Requires Grad: True
Layer: resnet.encoder.stages.2.layers.5.layer.0.normalization.weight | Size: torch.Size([256]) | Requires Grad: True
Layer: resnet.encoder.stages.2.layers.5.layer.0.normalization.bias | Size: torch.Size([256]) | Requires Grad: True
Layer: resnet.encoder.stages.2.layers.5.layer.1.convolution.weight | Size: torch.Size([256, 256, 3, 3]) | Requires Grad: True
Layer: resnet.encoder.stages.2.layers.5.layer.1.normalization.weight | Size: torch.Size([256]) | Requires Grad: True
Layer: resnet.encoder.stages.2.layers.5.layer.1.normalization.bias | Size: torch.Size([256]) | Requires Grad: True
Layer: resnet.encoder.stages.2.layers.5.layer.2.convolution.weight | Size: torch.Size([1024, 256, 1, 1]) | Requires Grad: True
Layer: resnet.encoder.stages.2.layers.5.layer.2.normalization.weight | Size: torch.Size([1024]) | Requires Grad: True
Layer: resnet.encoder.stages.2.layers.5.layer.2.normalization.bias | Size: torch.Size([1024]) | Requires Grad: True
Layer: resnet.encoder.stages.3.layers.0.shortcut.convolution.weight | Size: torch.Size([2048, 1024, 1, 1]) | Requires Grad: True
Layer: resnet.encoder.stages.3.layers.0.shortcut.normalization.weight | Size: torch.Size([2048]) | Requires Grad: True
Layer: resnet.encoder.stages.3.layers.0.shortcut.normalization.bias | Size: torch.Size([2048]) | Requires Grad: True
Layer: resnet.encoder.stages.3.layers.0.layer.0.convolution.weight | Size: torch.Size([512, 1024, 1, 1]) | Requires Grad: True
Layer: resnet.encoder.stages.3.layers.0.layer.0.normalization.weight | Size: torch.Size([512]) | Requires Grad: True
Layer: resnet.encoder.stages.3.layers.0.layer.0.normalization.bias | Size: torch.Size([512]) | Requires Grad: True
Layer: resnet.encoder.stages.3.layers.0.layer.1.convolution.weight | Size: torch.Size([512, 512, 3, 3]) | Requires Grad: True
Layer: resnet.encoder.stages.3.layers.0.layer.1.normalization.weight | Size: torch.Size([512]) | Requires Grad: True
Layer: resnet.encoder.stages.3.layers.0.layer.1.normalization.bias | Size: torch.Size([512]) | Requires Grad: True
Layer: resnet.encoder.stages.3.layers.0.layer.2.convolution.weight | Size: torch.Size([2048, 512, 1, 1]) | Requires Grad: True
Layer: resnet.encoder.stages.3.layers.0.layer.2.normalization.weight | Size: torch.Size([2048]) | Requires Grad: True
Layer: resnet.encoder.stages.3.layers.0.layer.2.normalization.bias | Size: torch.Size([2048]) | Requires Grad: True
Layer: resnet.encoder.stages.3.layers.1.layer.0.convolution.weight | Size: torch.Size([512, 2048, 1, 1]) | Requires Grad: True
Layer: resnet.encoder.stages.3.layers.1.layer.0.normalization.weight | Size: torch.Size([512]) | Requires Grad: True
Layer: resnet.encoder.stages.3.layers.1.layer.0.normalization.bias | Size: torch.Size([512]) | Requires Grad: True
Layer: resnet.encoder.stages.3.layers.1.layer.1.convolution.weight | Size: torch.Size([512, 512, 3, 3]) | Requires Grad: True
Layer: resnet.encoder.stages.3.layers.1.layer.1.normalization.weight | Size: torch.Size([512]) | Requires Grad: True
Layer: resnet.encoder.stages.3.layers.1.layer.1.normalization.bias | Size: torch.Size([512]) | Requires Grad: True
Layer: resnet.encoder.stages.3.layers.1.layer.2.convolution.weight | Size: torch.Size([2048, 512, 1, 1]) | Requires Grad: True
Layer: resnet.encoder.stages.3.layers.1.layer.2.normalization.weight | Size: torch.Size([2048]) | Requires Grad: True
Layer: resnet.encoder.stages.3.layers.1.layer.2.normalization.bias | Size: torch.Size([2048]) | Requires Grad: True
Layer: resnet.encoder.stages.3.layers.2.layer.0.convolution.weight | Size: torch.Size([512, 2048, 1, 1]) | Requires Grad: True
Layer: resnet.encoder.stages.3.layers.2.layer.0.normalization.weight | Size: torch.Size([512]) | Requires Grad: True
Layer: resnet.encoder.stages.3.layers.2.layer.0.normalization.bias | Size: torch.Size([512]) | Requires Grad: True
Layer: resnet.encoder.stages.3.layers.2.layer.1.convolution.weight | Size: torch.Size([512, 512, 3, 3]) | Requires Grad: True
Layer: resnet.encoder.stages.3.layers.2.layer.1.normalization.weight | Size: torch.Size([512]) | Requires Grad: True
Layer: resnet.encoder.stages.3.layers.2.layer.1.normalization.bias | Size: torch.Size([512]) | Requires Grad: True
Layer: resnet.encoder.stages.3.layers.2.layer.2.convolution.weight | Size: torch.Size([2048, 512, 1, 1]) | Requires Grad: True
Layer: resnet.encoder.stages.3.layers.2.layer.2.normalization.weight | Size: torch.Size([2048]) | Requires Grad: True
Layer: resnet.encoder.stages.3.layers.2.layer.2.normalization.bias | Size: torch.Size([2048]) | Requires Grad: True
Layer: classifier.1.weight | Size: torch.Size([2, 2048]) | Requires Grad: True
Layer: classifier.1.bias | Size: torch.Size([2]) | Requires Grad: True
</div>

