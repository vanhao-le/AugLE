import os
import pandas as pd
import shutil
import augly.image as imaugs
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
import augly.utils as utils
import torch.backends.cudnn as cudnn

cudnn.benchmark = True
# The augmentations we want to test robustness to
augmentations = {
    "1": imaugs.Brightness(factor=1.15),
    "2": imaugs.Brightness(factor=1.2),
    "3": imaugs.Brightness(factor=1.25),
    "4": imaugs.Brightness(factor=1.5),    
    "5": imaugs.Contrast(factor=1.8),
    "6": imaugs.Contrast(factor=1.5),
    "7": imaugs.Contrast(factor=0.8),
    "8": imaugs.Contrast(factor=0.9),
    "9": imaugs.Blur(radius=3),  
    "10": imaugs.Blur(radius=5),  
    "11": imaugs.RandomNoise(mean=0.1, var=0.05),
    "12": imaugs.RandomNoise(mean=0.05, var=0.05),
    "13": imaugs.RandomNoise(mean=0.07, var=0.05),
    "14": imaugs.RandomNoise(mean=0.05, var=0.02),
    "15": imaugs.RandomNoise(mean=0.01, var=0.1),
    "16": imaugs.RandomNoise(mean=0.15, var=0.05),
    "17": imaugs.RandomNoise(mean=0.2, var=0.05),
    "18": imaugs.Sharpen(factor=3),  
    "19": imaugs.Sharpen(factor=0.8),  
    "20": imaugs.Sharpen(factor=5),  
    "21": imaugs.ShufflePixels(factor=0.2),
    "22": imaugs.ShufflePixels(factor=0.1),
    "23": imaugs.ShufflePixels(factor=0.15),
    "24": imaugs.ShufflePixels(factor=0.21),
    "25": imaugs.EncodingQuality(quality=15),
    "26": imaugs.ChangeAspectRatio(ratio=3.0),
    "27": imaugs.ChangeAspectRatio(ratio=1.5),
    "28": imaugs.ChangeAspectRatio(ratio=0.5),
    "29": imaugs.ChangeAspectRatio(ratio=0.25),
    "30": imaugs.ChangeAspectRatio(ratio=0.1),
    "31": imaugs.ChangeAspectRatio(ratio=4),
    "32": imaugs.Opacity(level=0.8),
    "33": imaugs.Opacity(level=0.85),
    "34": imaugs.Opacity(level=0.75),
    "35": imaugs.Opacity(level=0.95),
    "36": imaugs.Opacity(level=0.9),
    "37": imaugs.Pixelization(ratio=0.15),
    "38": imaugs.Pixelization(ratio=0.1),
    "39": imaugs.Scale(factor=0.5),
    "40": imaugs.Scale(factor=2),
    "41": imaugs.ColorJitter(brightness_factor=1.6, contrast_factor=1.6, saturation_factor=1.6),
    "42": imaugs.ColorJitter(brightness_factor=0.75, contrast_factor=2.0, saturation_factor=0.3),
    "43": imaugs.ColorJitter(brightness_factor=2, contrast_factor=3, saturation_factor=4),
    "44": imaugs.ColorJitter(brightness_factor=1.5, contrast_factor=1.5, saturation_factor=1.5),
    "45": imaugs.ColorJitter(brightness_factor=1.5, contrast_factor=1.1, saturation_factor=2),
    "46": imaugs.HFlip(),
    "47": imaugs.HFlip(),
    "48": imaugs.HFlip(),
    "49": imaugs.PerspectiveTransform(),    
    "50": imaugs.PerspectiveTransform(sigma=50), 
    "51": imaugs.PerspectiveTransform(sigma=55), 
    "52": imaugs.PerspectiveTransform(sigma=60), 
    "53": imaugs.PerspectiveTransform(sigma=45), 
    "54": imaugs.Rotate(degrees=50),
    "55": imaugs.Rotate(degrees=10),
    "56": imaugs.Rotate(degrees=45),
    "57": imaugs.Rotate(degrees=100),
    "58": imaugs.Rotate(degrees=120),
    "59": imaugs.Rotate(degrees=150),
    "60": imaugs.Rotate(degrees=180),
    "61": imaugs.VFlip(),
    "62": imaugs.VFlip(),
    "63": imaugs.VFlip(),
    "64": imaugs.VFlip(),
    "65": imaugs.VFlip(),
    "66": imaugs.OverlayImage(Image.fromarray(np.random.rand(224, 224, 3).astype("uint8") * 255).convert("RGB"), opacity=0.7 ),
    "67": imaugs.OverlayImage(Image.fromarray(np.random.rand(224, 224, 3).astype("uint8") * 255).convert("RGB"), opacity=0.5 ),
    "68": imaugs.OverlayImage(Image.fromarray(np.random.rand(224, 224, 3).astype("uint8") * 255).convert("RGB"), opacity=0.3 ),
    "69": imaugs.OverlayImage(Image.fromarray(np.random.rand(224, 224, 3).astype("uint8") * 255).convert("RGB"), opacity=0.7, overlay_size=0.5),
    "70": imaugs.OverlayImage(Image.fromarray(np.random.rand(224, 224, 3).astype("uint8") * 255).convert("RGB"), opacity=0.7, overlay_size=0.25, x_pos=0.1, y_pos=0.1),
    "71": imaugs.OverlayImage(Image.fromarray(np.random.rand(224, 224, 3).astype("uint8") * 255).convert("RGB"), opacity=0.7, overlay_size=0.2, x_pos=0.5, y_pos=0.5),
    "72": imaugs.OverlayImage(Image.fromarray(np.random.rand(224, 224, 3).astype("uint8") * 255).convert("RGB"), opacity=0.7, overlay_size=0.3, x_pos=0.7, y_pos=0.8),
    "73": imaugs.OverlayImage(Image.fromarray(np.random.rand(224, 224, 3).astype("uint8") * 255).convert("RGB"), opacity=0.7, overlay_size=0.25, x_pos=0.1, y_pos=0.9),
    "74": imaugs.OverlayImage(Image.fromarray(np.random.rand(224, 224, 3).astype("uint8") * 255).convert("RGB"), opacity=0.8, overlay_size=0.7, x_pos=0.3, y_pos=0.4),
    "75": imaugs.OverlayOntoBackgroundImage(Image.fromarray(np.random.rand(224, 224, 3).astype("uint8") * 255).convert("RGB"), overlay_size=0.4),
    "76": imaugs.OverlayOntoBackgroundImage(Image.fromarray(np.random.rand(224, 224, 3).astype("uint8") * 255).convert("RGB"), overlay_size=0.8, x_pos=0.2, y_pos=0.4),
    "77": imaugs.OverlayOntoBackgroundImage(Image.fromarray(np.random.rand(224, 224, 3).astype("uint8") * 255).convert("RGB"), overlay_size=0.9, x_pos=0.1, y_pos=0.2),
    "78": imaugs.OverlayOntoBackgroundImage(Image.fromarray(np.random.rand(224, 224, 3).astype("uint8") * 255).convert("RGB"), overlay_size=0.5, x_pos=0.1, y_pos=0.4),
    "79": imaugs.OverlayOntoBackgroundImage(Image.fromarray(np.random.rand(224, 224, 3).astype("uint8") * 255).convert("RGB"), overlay_size=0.5, x_pos=0.1, y_pos=0.35),
    "80": imaugs.OverlayOntoBackgroundImage(Image.fromarray(np.random.rand(224, 224, 3).astype("uint8") * 255).convert("RGB"), overlay_size=0.5, x_pos=0.15, y_pos=0.55),
    "81": imaugs.OverlayImage(Image.fromarray(np.random.rand(224, 224, 3).astype("uint8") * 255).convert("RGB"), overlay_size=0.5),
    "82": imaugs.OverlayEmoji(y_pos=0.3, emoji_size=0.8),
    "83": imaugs.OverlayEmoji(y_pos=0.1, emoji_size=0.2),
    "84": imaugs.OverlayEmoji(y_pos=0.2, emoji_size=0.25),
    "85": imaugs.OverlayEmoji(x_pos=0.7, y_pos=0.1, emoji_size=0.25),
    "86": imaugs.OverlayEmoji(x_pos=0.7, y_pos=0.7, emoji_size=0.3),
    "87": imaugs.OverlayEmoji(x_pos=0.1, y_pos=0.4, emoji_size=0.3),
    "88": imaugs.OverlayEmoji(x_pos=0.0, y_pos=0.5, emoji_size=0.3),
    "89": imaugs.OverlayEmoji(x_pos=0.5, y_pos=0.5, emoji_size=0.3),
    "90": imaugs.OverlayEmoji(y_pos=0.3, emoji_size=0.5),
    "91": imaugs.OverlayText(font_size=0.5, x_pos=0.2, y_pos=0.1),
    "92": imaugs.OverlayText(font_size=0.25, x_pos=0.1, y_pos=0.3),
    "93": imaugs.OverlayText(font_size=0.5, x_pos=0.3, y_pos=0.5),
    "94": imaugs.OverlayText(font_size=0.5, x_pos=0.3, y_pos=0.5),
    "95": imaugs.OverlayText(font_size=0.5, color=(125, 125, 125), x_pos=0.3, y_pos=0.5),
    "96": imaugs.OverlayText(font_size=0.5, color=(0, 255, 255), x_pos=0.1, y_pos=0.5),
    "97": imaugs.OverlayText(font_size=0.5, color=(0, 255, 255), x_pos=0.1, y_pos=0.1),
    "98": imaugs.OverlayText(font_size=0.8, x_pos=0.3, y_pos=0.4),
    "99": imaugs.OverlayOntoScreenshot(),
    "100": imaugs.OverlayOntoScreenshot(),
    "101": imaugs.OverlayOntoScreenshot(),
    "102": imaugs.OverlayOntoScreenshot(template_filepath=os.path.join(utils.SCREENSHOT_TEMPLATES_DIR, "mobile.png")),
    "103": imaugs.OverlayOntoScreenshot(template_filepath=os.path.join(utils.SCREENSHOT_TEMPLATES_DIR, "mobile.png")),
    "104": imaugs.OverlayOntoScreenshot(template_filepath=os.path.join(utils.SCREENSHOT_TEMPLATES_DIR, "mobile.png")),
    "105": imaugs.OverlayOntoScreenshot(template_filepath=os.path.join(utils.SCREENSHOT_TEMPLATES_DIR, "mobile.png")),
    "106": imaugs.OverlayOntoScreenshot(template_filepath=os.path.join(utils.SCREENSHOT_TEMPLATES_DIR, "mobile.png")),
    "107": imaugs.OverlayOntoScreenshot(template_filepath=os.path.join(utils.SCREENSHOT_TEMPLATES_DIR, "mobile.png")),
    "108": imaugs.OverlayOntoScreenshot(template_filepath=os.path.join(utils.SCREENSHOT_TEMPLATES_DIR, "mobile.png")),
    "109": imaugs.OverlayStripes(line_angle=-30, line_density=0.2, line_width=0.4, line_type="dashed"),
    "110": imaugs.OverlayStripes(line_angle=90, line_density=0.3, line_width=0.1, line_type="solid"),
    "111": imaugs.OverlayStripes(line_density=0.3, line_width=0.1, line_type="solid", line_opacity=0.5, line_color=(120, 0, 200), line_angle=25.0),
    "112": imaugs.OverlayStripes(line_type="dashed", line_opacity=0.5, line_color=(120, 0, 200), line_angle=25.0),
    "113": imaugs.OverlayStripes(line_type="dashed", line_opacity=0.25, line_color=(120, 125, 200), line_angle=120.0),
    "114": imaugs.OverlayStripes(line_type='dashed', line_opacity=0.55, line_color=(0, 255, 255), line_angle=90),
    "115": imaugs.OverlayStripes(line_type="dashed", line_opacity=0.6, line_color=(120, 125, 200), line_angle=90),
    "116": imaugs.MemeFormat(),
    "117": imaugs.MemeFormat(text_color=(120,125, 200)),
    "118": imaugs.MemeFormat(text_color=(0,255, 200)),
    "119": imaugs.MemeFormat(text_color=(0,255, 200), caption_height=100),
    "120": imaugs.MemeFormat(text_color=(120,155, 200), caption_height=100),
    "121": imaugs.PerspectiveTransform(sigma=30),
    "122": imaugs.PerspectiveTransform(sigma=80),    
    "123": imaugs.Pad(w_factor=0.15, h_factor=0.2, color=(np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255))),
    "124": imaugs.Pad(w_factor=0.15, h_factor=0.25, color=(np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255))),
    "125": imaugs.Pad(w_factor=0.1, h_factor=0.3, color=(np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255))),
    "126": imaugs.Pad(w_factor=0.1, h_factor=0.2, color=(np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255))),
    "127": imaugs.Pad(w_factor=0.11, h_factor=0.21, color=(np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255))),
    "128": imaugs.Pad(w_factor=0.5, h_factor=0.25, color=(np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255))),
    "129": imaugs.Pad(w_factor=0.3, h_factor=0.1, color=(np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255))),
    "130": imaugs.PadSquare(color=(np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255))),
    "131": imaugs.PadSquare(color=(np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255))),
    "132": imaugs.PadSquare(color=(np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255))),
    "133": imaugs.PadSquare(color=(np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255))),
    "134": imaugs.PadSquare(color=(np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255))),
    "135": imaugs.PadSquare(color=(np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255))),
    "136": imaugs.PadSquare(color=(np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255))),
    "137": imaugs.PadSquare(color=(np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255))),
    "138": imaugs.PadSquare(color=(np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255))),
    "139": imaugs.PadSquare(color=(np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255))),
    "140": imaugs.PadSquare(color=(np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255))),
    "141": imaugs.PadSquare(color=(np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255))),
    "142": imaugs.Grayscale(),
    "143": imaugs.Grayscale(),
    "144": imaugs.Grayscale(),
    "145": imaugs.Grayscale(),
    "146": imaugs.OverlayText(font_size=0.3, color=(np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255)), x_pos= np.random.rand(), y_pos= np.random.rand()),
    "147": imaugs.OverlayText(font_size=0.1, color=(np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255)), x_pos= np.random.rand(), y_pos= np.random.rand()),
    "148": imaugs.OverlayText(font_size=0.25, color=(np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255)), x_pos= np.random.rand(), y_pos= np.random.rand()),
    "149": imaugs.OverlayEmoji(opacity= np.random.rand(), emoji_size= 0.25, x_pos= np.random.rand(), y_pos= np.random.rand()),  
    "150": imaugs.OverlayEmoji(opacity= np.random.rand(), emoji_size= 0.25, x_pos= np.random.rand(), y_pos= np.random.rand()),
    "151": imaugs.OverlayEmoji(opacity= np.random.rand(), emoji_size= 0.25, x_pos= np.random.rand(), y_pos= np.random.rand()),
    "152": imaugs.OverlayOntoScreenshot(),
    "153": imaugs.OverlayOntoScreenshot(),
    "154": imaugs.OverlayOntoScreenshot(),
    "155": imaugs.OverlayOntoScreenshot(),
    "156": imaugs.OverlayOntoScreenshot(),
    "157": imaugs.OverlayOntoScreenshot(),
    "158": imaugs.OverlayOntoScreenshot(),
    "159": imaugs.OverlayOntoScreenshot(),

}

# Processing transformations which will be applied to all images
base_transforms = [    
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
]

BATCH_SIZE = 256

class MyDataset(data.Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
        self.file_list = list()
        for f in os.listdir(self.image_paths):
            file_name, file_ext = os.path.splitext(f)            
            if file_ext == '.jpg':
                self.file_list.append(f)
        

    def __getitem__(self, index):
        input_image_path = os.path.join(self.image_paths, self.file_list[index])
        image = Image.open(input_image_path)  
        if self.transform:
            img = self.transform(image)
        
        return img, self.file_list[index]      

    def __len__(self):
        return len(self.file_list)

def data_augmentation():
    reference_csv = r'data\category.csv'
    db_root = r'D:\VSC\train\reference'
    aug_path = r'D:\VSC\train\aug'
    image_path = r'D:\VSC\train\frame'
    

    df = pd.read_csv(reference_csv)
    
    data = []
    
    for item in df.itertuples():
        video_id = item.class_name
        class_id = item.classIDx

        count = 1
        for aug_name, aug in augmentations.items():      
        
            new_id = video_id + '-' + str(count)
            ouput_path = os.path.join(aug_path, new_id)
            if not os.path.exists(ouput_path):
                os.makedirs(ouput_path)            

            input_path = os.path.join(image_path, video_id)

            aug_transform = transforms.Compose([aug] + base_transforms)

            input_dataset =  MyDataset(image_paths=input_path, transform=aug_transform)            
            input_loader = DataLoader(input_dataset, batch_size=BATCH_SIZE, shuffle=False)
            
            for images, names in input_loader:
                for i in range(0, len(names)):
                    output_image_path = os.path.join(ouput_path, str(names[i]))                
                    save_image(images[i], output_image_path)                        
            
            case = {'video_id': new_id, 'classIDx': class_id}
            data.append(case)
            count += 1
              

    df = pd.DataFrame(data)
    df.to_csv(r'data\augly-100.csv', index=False)
    

def main():
    data_augmentation()  

if __name__ == '__main__':
    main()
    
