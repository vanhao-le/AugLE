# Note: restart runtime after this import before running the augmentations
# pip install -U augly[image]
# conda install -c conda-forge python-magic
'''
In the case, we got an error with python-magic. Open CMD and try below commands (Windows OS) because of
You'll need DLLs for libmagic. @julian-r maintains a pypi package with the DLLs, you can fetch it with:
pip uninstall python-magic 
pip install python-magic-bin
'''

import os
import augly.image as imaugs
from PIL import Image



'''
using show() and save() function if we want to display or save image. 
The AugLy also provides the output_path paramter in every function for saving the image.

input_img.show()
input_img = input_img.save(output_path)
'''


image_path = r"tmp\meteo.png"
output_path = r"output\meteo_aug.png"

'''
Function-based
'''

# Augmentation functions can accept image paths as input and
# always return the resulting augmented PIL Image

aug_image = imaugs.overlay_emoji(image_path, opacity=1.0, emoji_size=0.15)

# Augmentation functions can also accept PIL Images as input

aug_image = imaugs.pad_square(aug_image)

# If an output path is specified, the image will also be saved to a file

aug_image = imaugs.overlay_onto_screenshot(aug_image)

# Adding staturation

img_pil = Image.open(image_path)
aug_saturation = imaugs.Saturation(factor=5.0)
img_stat = aug_saturation(img_pil)
# img_stat.show()

# overlaying stripes and the transformed image

img_aug = imaugs.overlay_stripes(image_path, line_type='dashed', line_opacity=0.5, line_color=(120, 0, 200), line_angle=25.0)
# img_aug.show()

import augly.utils as utils

img_pil = Image.open(image_path)
aug_overlay = imaugs.OverlayOntoScreenshot(
    template_filepath=os.path.join(utils.SCREENSHOT_TEMPLATES_DIR, "mobile.png")
    )
img_aug = aug_overlay(img_pil)

img_aug.show()


# scale augmentation

input_img = imaugs.scale(image_path, factor=0.2)

# Now we can apply various augmentations to the scaled image

input_img= imaugs.meme_format(
    image_path,  caption_height=75, meme_bg_color=(0, 0, 0), text_color=(255, 255, 255),
    )

"""
You can optionally pass in a metadata list, to which metadata about the
augmentation will be appended, including kwargs, input & output
dimensions, and intensity (defined based on the kwargs for each
augmentation).
"""
meta = []
input_img = imaugs.shuffle_pixels(image_path, factor=0.3, metadata=meta)
print(meta)

"""
You can also pass in bounding boxes, which will be transformed along with
the image & included in the metadata (note: you must provide metadata to
get the transformed bboxes)
"""

meta = []
input_img = imaugs.rotate(
    image_path, degrees=15, metadata=meta, bboxes=[(20, 6, 250, 180)], bbox_format="pascal_voc",
    )
# input_img.show()
print(meta)


# For all the augmentations, we have class-based definitions as well as functional

meta = []
aug = imaugs.PerspectiveTransform(sigma=20.0)
input_img = Image.open(image_path)
input_img =  aug(input_img, metadata=meta)

# input_img.show()
print(meta)


"""
For some augmentations, we also provide versions that will randomly sample
from a set of parameters (e.g. for ChangeAspectRatio, RandomAspectRatio
samples an emoji from Twitter's Twemoji set which we provide in the augly
package). The metadata will contain the actual sampled param values.
"""
meta = []
aug = imaugs.RandomAspectRatio()
input_img = Image.open(image_path)
input_img =  aug(input_img, metadata=meta)
# input_img.show()
print(meta)


'''
Class-based

'''
# AugLy also integrates seamlessly with PyTorch transforms
# Note: you must have torchvision installed, which it is by default in colab

import torchvision.transforms as transforms

COLOR_JITTER_PARAMS = {
    "brightness_factor": 1.2,
    "contrast_factor": 1.2,
    "saturation_factor": 1.4,
}

AUGMENTATIONS = [
    imaugs.Blur(),
    imaugs.ColorJitter(**COLOR_JITTER_PARAMS),
    imaugs.OneOf(
        [imaugs.OverlayOntoScreenshot(), imaugs.OverlayEmoji(), imaugs.OverlayText()]
    ),
]

TRANSFORMS = imaugs.Compose(AUGMENTATIONS)
TENSOR_TRANSFORMS = transforms.Compose(AUGMENTATIONS + [transforms.ToTensor()])

# aug_image is a PIL image with your augs applied!
# aug_tensor_image is a Tensor with your augs applied!
image = Image.open(image_path)
aug_image = TRANSFORMS(image)
aug_tensor_image = TENSOR_TRANSFORMS(image)

# aug_image.show()

'''
NumPy wrapper

'''

# We also provide a numpy wrapper in case your data is in np.ndarray format
import numpy as np
from augly.image import aug_np_wrapper, overlay_emoji

np_image = np.zeros((300, 300))
# pass in function arguments as kwargs
np_aug_img = aug_np_wrapper(np_image, overlay_emoji, **{'opacity': 0.5, 'y_pos': 0.45})