import os
import argparse
parser = argparse.ArgumentParser(description='Train a CNN with Hebbian learning on CIFAR-100')
parser.add_argument('--cuda', type=str, default='0', help='CUDA device (default: 0)')
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 2000)')
parser.add_argument('--noise_mean', type=float, default=0.0, help='the mean of the added noise')
parser.add_argument('--noise_std', type=float, default=0.0,help='the std of the added noise')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
import numpy as np
import random
import matplotlib.pyplot as plt 
from PIL import Image
import math

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
class SoftHebbConv2d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = 0,
            dilation: int = 1,
            groups: int = 1,
            t_invert: float = 12,
    ) -> None:
        """
        Simplified implementation of Conv2d learnt with SoftHebb; an unsupervised, efficient and bio-plausible
        learning algorithm.
        This simplified implementation omits certain configurable aspects, like using a bias, groups>1, etc. which can
        be found in the full implementation in hebbconv.py
        """
        super(SoftHebbConv2d, self).__init__()
        assert groups == 1, "Simple implementation does not support groups > 1."
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.padding_mode = 'reflect'
        self.F_padding = (padding, padding, padding, padding)
        weight_range = 25 / math.sqrt((in_channels / groups) * kernel_size * kernel_size)
        self.weight = nn.Parameter(weight_range * torch.randn((out_channels, in_channels // groups, *self.kernel_size)))
        self.t_invert = torch.tensor(t_invert)

    def forward(self, x):
        x = F.pad(x, self.F_padding, self.padding_mode)  # pad input
        # perform conv, obtain weighted input u \in [B, OC, OH, OW]
        weighted_input = F.conv2d(x, self.weight, None, self.stride, 0, self.dilation, self.groups)

        return weighted_input
    
class Triangle(nn.Module):
    def __init__(self, power: float = 1, inplace: bool = True):
        super(Triangle, self).__init__()
        self.inplace = inplace
        self.power = power

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = input - torch.mean(input.data, axis=1, keepdims=True)
        return F.relu(input, inplace=self.inplace) ** self.power
    
class DeepSoftHebb(nn.Module):
    def __init__(self):
        super(DeepSoftHebb, self).__init__()
        # block 1
        self.bn1 = nn.BatchNorm2d(3, affine=False)
        self.conv1 = SoftHebbConv2d(in_channels=3, out_channels=384, kernel_size=5, padding=2)
        self.activ1 = Triangle(power=0.7)
        self.pool1 = nn.MaxPool2d(kernel_size=4, stride=2, padding=1)
        # block 2
        self.bn2 = nn.BatchNorm2d(384, affine=False)
        self.conv2 = SoftHebbConv2d(in_channels=384, out_channels=768, kernel_size=3, padding=1)
        self.activ2 = Triangle(power=1.4)
        self.pool2 = nn.MaxPool2d(kernel_size=4, stride=2, padding=1)
        # block 3
        self.bn3 = nn.BatchNorm2d(768, affine=False)
        self.conv3 = SoftHebbConv2d(in_channels=768, out_channels=1536, kernel_size=3, padding=1)
        self.activ3 = Triangle(power=1.)
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)


    def forward(self, x):
        # block 1
        out = self.pool1(self.activ1(self.conv1(self.bn1(x))))
        # block 2
        out = self.pool2(self.activ2(self.conv2(self.bn2(out))))
        # block 3
        out = self.pool3(self.activ3(self.conv3(self.bn3(out))))
        return out
    
    
class HebbLayer(nn.Module):
    def __init__(self, block, in_channel, out_channel, proj_num, label_num=100, res=False):
        super(HebbLayer, self).__init__()
        self.block = block
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.res = res
        self.criteria = nn.MSELoss()
        self.out_proj = nn.Sequential(
            nn.Conv2d(out_channel, out_channel // 2, 1, 1, 0),
            nn.BatchNorm2d(out_channel // 2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(out_channel // 2, proj_num),
        )
        self.img_proj = nn.Sequential(
            nn.Linear(label_num, proj_num),
        )

    def forward(self, x, z, img):
        y = self.block(x)
        loss = 0
        if self.res:
            if self.in_channel != self.out_channel:
                x = F.pad(x, (0, 0, 0, 0, 0, self.out_channel - self.in_channel))
                x = nn.AvgPool2d(2, 2)(x)
            y = y + x

        return y, loss

class CNN(nn.Module):
    def __init__(self, label_num=10):
        super(CNN, self).__init__()
        self.proj_num = 256
        # self.channel = 256
        self.channel = 1536

        self.block1 = HebbLayer(
            nn.Sequential(
                nn.Conv2d(3, self.channel // 4, 3, 1, 1),
                nn.BatchNorm2d(self.channel // 4),
                nn.LeakyReLU(0.2),
                nn.MaxPool2d(4, 2, 1),
            ),
            3, self.channel // 4, self.proj_num, res=False
        )
        self.block2 = HebbLayer(
            nn.Sequential(
                nn.Conv2d(self.channel // 4, self.channel // 2, 3, 1, 1),
                nn.BatchNorm2d(self.channel // 2),
                nn.LeakyReLU(0.2),
                nn.MaxPool2d(4, 2, 1),
            ),
            self.channel // 4, self.channel // 2, self.proj_num, res=False
        )
        self.block3 = HebbLayer(
            nn.Sequential(
                nn.Conv2d(self.channel // 2, self.channel, 3, 1, 1),
                nn.BatchNorm2d(self.channel),
                nn.LeakyReLU(0.2),
                nn.MaxPool2d(4, 2, 1),
            ),
            self.channel // 2, self.channel, self.proj_num, res=True
        )


    def forward(self, x, z=None, detachx=True):
        loss = 0
        img = x.clone()
        x, loss1 = self.block1(x, z, img)
        loss += loss1
        x, loss2 = self.block2(x, z, img)
        loss += loss2
        x, loss3 = self.block3(x, z, img)

        return x
    
class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Input should be a torch.Tensor, got {type(tensor)}")
        return tensor + torch.randn(tensor.size(), device=tensor.device) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


noise = AddGaussianNoise(mean=args.noise_mean, std=args.noise_std)

seed_all(args.seed)


# Load the state dictionary
encoder = []

encoder.append(DeepSoftHebb().cuda())
encoder.append(DeepSoftHebb().cuda())
encoder.append(CNN().cuda())
encoder.append(CNN().cuda())

encoder[0].load_state_dict(torch.load('./models/softhebb.pth'), strict=False)
encoder[1].load_state_dict(torch.load('./models/softhebb_100.pth'), strict=False)
encoder[2].load_state_dict(torch.load('./models/bp.pth'), strict=False)
encoder[3].load_state_dict(torch.load('./models/hebb.pth'), strict=False)

name = ['SoftHebb', 'SoftHebb(100 epochs)', 'BP', 'SPHeRe']

class UpsampleDecoder(nn.Module):
    def __init__(self):
        super(UpsampleDecoder, self).__init__()

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1536, 512, kernel_size=3, stride=2, padding=1, output_padding=1),  # (1, 512, 8, 8)
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),  # (1, 256, 16, 16)
            nn.ReLU(),
            nn.ConvTranspose2d(256, 3, kernel_size=3, stride=2, padding=1, output_padding=1),  # (1, 3, 32, 32)
            # nn.ReLU(),
            # nn.Conv2d(128, 3, kernel_size=3, padding=1),  # Reduce channels to 3
        )
        
    def forward(self, x):
        return self.decoder(x)
    
decoder = []
for i in range(4):
    decoder.append(UpsampleDecoder().cuda())



# Load the image
image_path = './reconstruction/samples/sample.png'
image = Image.open(image_path).convert('RGB')

# Define a transform to convert the image to a tensor
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Apply the transform to the image
image_tensor = transform(image).unsqueeze(0).cuda()  # Add batch dimension and move to GPU
background = torch.zeros_like(image_tensor)
original = image_tensor.clone()
image_tensor = noise(image_tensor)

print(f'Added Noise: mean: {args.noise_mean}, std: {args.noise_std}')


print('loss of noise:', F.mse_loss(image_tensor, original).item()*1000)

losses = []
dec= []

for j in range(4):
    optimizer = optim.Adam(decoder[j].parameters(), lr=1e-3)
    for i in range(100):
        loss = 0
        encoded = encoder[j](image_tensor)
        decoded = decoder[j](encoded)
        loss = F.mse_loss(decoded, original)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    losses.append(loss.item()*1000)
    dec.append(decoded)
    print(f'Finished training decoder {j+1} out of 4')

print('\nLosses:')
print('\n'.join(f'{name[i]:<20}: {losses[i]:.6f}' for i in range(4)))

# Move the image to CPU and convert to numpy
image_tensor = image_tensor.cpu().numpy()
original = original.cpu().numpy()
decoded_images = [img.detach().cpu().numpy() for img in dec]

# Ensure the tensor is in the correct shape (C, H, W)
original_image = original.squeeze().transpose(1, 2, 0)
image_tensor = image_tensor.squeeze().transpose(1, 2, 0)

# Normalize original image to 0-1 range if needed
if original_image.min() < 0 or original_image.max() > 1:
    original_image = (original_image - original_image.min()) / (original_image.max() - original_image.min())
    
if image_tensor.min() < 0 or image_tensor.max() > 1:
    image_tensor = (image_tensor - image_tensor.min()) / (image_tensor.max() - image_tensor.min())


fig, axes = plt.subplots(1, 5, figsize=(20, 5))
# Original image
axes[0].imshow(original_image)  # Convert to (H, W, C) format
axes[0].set_title('Original Image')
axes[0].axis('off')

# Decoded images
for i, decoded_image in enumerate(decoded_images):
    decoded_image = decoded_image.squeeze().transpose(1, 2, 0)
    # Normalize image to 0-1 range if needed
    if decoded_image.min() < 0 or decoded_image.max() > 1:
        decoded_image = (decoded_image - decoded_image.min()) / (decoded_image.max() - decoded_image.min())
    axes[i + 1].imshow(decoded_image)  # Convert to (H, W, C) format
    axes[i + 1].set_title(name[i])
    axes[i + 1].axis('off')
    
captions = ['Loss = 0'] + [f'Loss = {losses[i]:.3f}' for i in range(len(decoded_images))]
for i, (ax, caption) in enumerate(zip(axes, captions)):
    bbox = ax.get_position()  # Get bounding box of subplot
    x_pos = (bbox.x0 + bbox.x1) / 2  # Get center x position
    fig.text(x_pos, bbox.y0 - 0.1, caption, ha='center', fontsize=12)  # Adjust y position


# Save the figure
plt.savefig('figures/output.png')