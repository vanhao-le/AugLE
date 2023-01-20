# Note: restart runtime after this import before running the augmentations
# pip install augly[av]
# conda install -c conda-forge python-magic
'''
In the case, we got an error with python-magic. Try below commands (Windows OS) because of
You'll need DLLs for libmagic. @julian-r maintains a pypi package with the DLLs, you can fetch it with:
pip uninstall python-magic 
pip install python-magic-bin

Installing ffmpeg is only needed for the video module of augly
If you're using conda you can do this with:
conda install -c conda-forge ffmpeg
If you aren’t using conda, you can run:
!sudo add-apt-repository ppa:jonathonf/ffmpeg-4
!apt install ffmpeg

Additional:

1. High-performance cross-platform Video Processing Python framework powerpacked with unique trailblazing features.

pip install vidgear

2. A python package for music and audio analysis.

pip install librosa

'''

import augly.video as vidaugs


input_vid_path =  r"tmp\dance.mp4"
out_vid_path =  r"output\dance.mp4"

'''
Function-based

'''

# output_path will be overwritten

# vidaugs.rotate(input_vid_path, out_vid_path , degrees=90)  


# We can use the AugLy trim augmentation, and save the trimmed video

# vidaugs.trim(input_vid_path, output_path=out_vid_path, start=0, end=3)


# We can apply various augmentations to the trimmed video
# output_path will be overwritten

# vidaugs.overlay_text(input_vid_path, out_vid_path)


"""
You can optionally pass in a metadata list, to which metadata about the
augmentation will be appended, including kwargs, input & output dimensions,
the matching segments in the input & output videos (useful in case of temporal
editing), and intensity (defined based on the kwargs for each augmentation).
"""
# meta = []
# vidaugs.loop(input_vid_path, out_vid_path, num_loops=1, metadata=meta,)
# print(meta)


# For all the augmentations, we have class-based definitions as well as functional

# meta = []
# aug = vidaugs.InsertInBackground()
# aug(input_vid_path, out_vid_path, metadata=meta)
# print(meta)


"""
For some augmentations, we also provide versions that will randomly sample
from a set of parameters (e.g. for OverlayEmoji, RandomEmojiOverlay samples
an emoji from Twitter's Twemoji set which we provide in the augly package).
The metadata will contain the actual sampled param values.
"""

# meta = []
# aug = vidaugs.RandomEmojiOverlay()
# aug(input_vid_path, out_vid_path, metadata=meta)
# print(meta)


'''
Class-based

'''

COLOR_JITTER_PARAMS = {
    "brightness_factor": 0.15,
    "contrast_factor": 1.3,
    "saturation_factor": 2.0,
}


AUGMENTATIONS = [
    vidaugs.ColorJitter(**COLOR_JITTER_PARAMS),
    vidaugs.HFlip(),
    vidaugs.OneOf(
        [
            vidaugs.RandomEmojiOverlay(),           
            vidaugs.Shift(x_factor=0.25, y_factor=0.25),
        ]
    ),
]

TRANSFORMS = vidaugs.Compose(AUGMENTATIONS)

# transformed video now stored in `out_video_path`

TRANSFORMS(input_vid_path, out_vid_path)  