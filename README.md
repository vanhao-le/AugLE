# AugLE
This repos includes and explains some common methods for image and video augmentations using the AugLy library of Meta AI. Data augmentations are vital to ensure robustness of AI models. If we can teach our models to be robust to perturbations of unimportant attributes of data, models will learn to focus on the important attributes of data for a particular use case.

The AugLy library composes 4 modalities (audio, image, text and video). However, I will just want to apply two of them (i.e., image and video) in this repos. The complete source and document can be found in the following Links section.

# Installation

Install image and video sub-library as opposed to all.

```python
pip install augly[image]

pip install augly[video]
```

In some environments, pip doesn't install python-magic as expected. In that case, you will need to additionally run:

```
conda install -c conda-forge python-magic
```

# Links

1. Facebook AI blog post: https://ai.facebook.com/blog/augly-a-new-data-augmentation-library-to-help-build-more-robust-ai-models/
2. PyPi package: https://pypi.org/project/augly/
3. Arxiv paper: https://arxiv.org/abs/2201.06494
4. Examples: https://github.com/facebookresearch/AugLy/tree/main/examples
5. Document: https://augly.readthedocs.io/en/latest/ 
