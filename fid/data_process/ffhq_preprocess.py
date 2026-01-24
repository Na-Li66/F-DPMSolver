import os
import fnmatch
import numpy as np
from PIL import Image
import albumentations as A
from torch.utils.data import Dataset

class ImagePaths(Dataset):
    def __init__(self, paths, size=None, random_crop=False, labels=None):
        assert isinstance(paths, (list, tuple)) and len(paths) > 0, "paths need empty list"
        self.size = int(size) if size is not None else None
        self.random_crop = bool(random_crop)
        self.labels = {} if labels is None else dict(labels)
        self.labels["file_path_"] = list(paths)
        self._length = len(paths)

        if self.size is not None and self.size > 0:
            self.rescaler = A.SmallestMaxSize(max_size=self.size)
            self.cropper = (
                A.RandomCrop(height=self.size, width=self.size)
                if self.random_crop else
                A.CenterCrop(height=self.size, width=self.size)
            )
            self.preprocessor = A.Compose([self.rescaler, self.cropper])
        else:
            self.preprocessor = lambda **kwargs: kwargs

    def __len__(self):
        return self._length

    def preprocess_image(self, image_path: str) -> np.ndarray:
        try:
            with Image.open(image_path) as img:
                if img.mode != "RGB":
                    img = img.convert("RGB")
                image = np.array(img).astype(np.uint8)
        except Exception as e:
            raise RuntimeError(f"Fail: {image_path}. Reason: {e}")

        image = self.preprocessor(image=image)["image"]
        image = (image / 127.5 - 1.0).astype(np.float32)
        return image

    def __getitem__(self, i: int):
        if i < 0 or i >= self._length:
            raise IndexError(f"Over-length index: {i} (on length {self._length})")
        example = {"image": self.preprocess_image(self.labels["file_path_"][i])}
        for k, v in self.labels.items():
            example[k] = v[i]
        return example

def list_images_in_dir(
    source_directory: str,
    patterns=("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"),
    recursive: bool = True,
):
    source_directory = os.path.abspath(source_directory)
    paths = []
    if recursive:
        for root, _, files in os.walk(source_directory):
            for pat in patterns:
                for name in fnmatch.filter(files, pat):
                    paths.append(os.path.join(root, name))
    else:
        files = os.listdir(source_directory)
        for pat in patterns:
            for name in fnmatch.filter(files, pat):
                paths.append(os.path.join(source_directory, name))
    paths = sorted(set(paths))
    if not paths:
        raise ValueError(f"Not find any picture in {source_directory}")
    return paths


def preprocess_and_copy_all_images(
    source_directory: str,
    new_target_directory: str,
    image_size: int,
    random_crop: bool = False,
    recursive: bool = True,
    keep_structure: bool = False,
):
    image_paths = list_images_in_dir(source_directory, recursive=recursive)
    dataset = ImagePaths(paths=image_paths, size=image_size, random_crop=random_crop)

    os.makedirs(new_target_directory, exist_ok=True)

    src_root = os.path.abspath(source_directory)

    for i in range(len(dataset)):
        sample = dataset[i]
        pre_img = sample["image"]  # [-1,1], HWC
        out = np.clip(pre_img * 127.5 + 127.5, 0, 255).astype(np.uint8)

        src_path = sample["file_path_"]
        if keep_structure:
            rel = os.path.relpath(src_path, src_root)
            save_path = os.path.join(new_target_directory, rel)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        else:
            save_path = os.path.join(new_target_directory, os.path.basename(src_path))

        Image.fromarray(out).save(save_path)

    print(f"Total {len(dataset)} pictures, saved in {new_target_directory}")


if __name__ == "__main__":
    preprocess_and_copy_all_images(
        source_directory="data/FFHQ/1024",
        new_target_directory="data/FFHQ/256",
        image_size=256,
        random_crop=False,
        recursive=True,
        keep_structure=True
    )
