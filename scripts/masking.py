from argparse import ArgumentParser
from pathlib import Path
from PIL import Image
from tqdm import tqdm

import os
import torch
from torchvision import transforms

from birefnet.models.birefnet import BiRefNet
from birefnet.utils import check_state_dict


def load_birefnet_from_local_weights(weight_path):
    # BiRefNet_HR-general-epoch_130.pth
    birefnet = BiRefNet(bb_pretrained=False)
    state_dict = torch.load(str(weight_path), map_location='cpu')
    state_dict = check_state_dict(state_dict)
    birefnet.load_state_dict(state_dict)
    return birefnet


def extract_mask(birefnet, image_path, tf_size=(1024, 1024), out_size=None):
    # Data settings
    transform_image = transforms.Compose([
        transforms.Resize(tf_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert("RGB")
    input_images = transform_image(image).unsqueeze(0).to('cuda').half()

    # Prediction
    with torch.no_grad():
        preds = birefnet(input_images)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(out_size if out_size else image.size)
    return mask


if __name__ == "__main__":
    parser = ArgumentParser("BiRefNet masking")
    parser.add_argument("-i", "--images", type=str, required=True)
    parser.add_argument("-o", "--output", type=str, default="", help="Output dir, default to the same place of images")
    parser.add_argument("-w", "--weight", type=str, default="", help="BiRefNet checkpoint, empty to load from HF")
    args = parser.parse_args()

    if not args.weight:
        # Load weights from Hugging Face Models
        birefnet = BiRefNet.from_pretrained('ZhengPeng7/BiRefNet')
    else:
        weight_file = Path(args.weight)
        if not weight_file.exists():
            raise FileNotFoundError(str(weight_file))
        birefnet = load_birefnet_from_local_weights(weight_file.resolve())

    images_dir = Path(args.images)
    output_dir = images_dir.parent / "masks" if not args.output else Path(args.output)
    os.makedirs(output_dir, exist_ok=True)

    torch.set_float32_matmul_precision(['high', 'highest'][0])
    birefnet.to('cuda')
    birefnet.eval()
    birefnet.half()

    image_files = [f for f in images_dir.iterdir() if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    for image_file in tqdm(image_files, ncols=64, desc="Masking"):
        mask = extract_mask(birefnet, image_file)
        mask.save(output_dir / f"{image_file.stem}.png")
