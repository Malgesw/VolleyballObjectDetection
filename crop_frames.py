import os
import cv2
import argparse


def resize_images(input_dir, output_dir, width=640, height=480):
    os.makedirs(output_dir, exist_ok=True)
    for fname in os.listdir(input_dir):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
            continue
        src_path = os.path.join(input_dir, fname)
        img = cv2.imread(src_path)
        if img is None:
            print(f"Failed to read image: {src_path}")
            continue
        resized = cv2.resize(img, (width, height))
        dst_path = os.path.join(output_dir, fname)
        cv2.imwrite(dst_path, resized)
    print(f"Resized images saved to {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Batch-resize images to 480p')
    parser.add_argument('--input_dir', '-i', required=True,
                        help='Directory of input images')
    parser.add_argument('--output_dir', '-o', required=True,
                        help='Directory for resized output')
    args = parser.parse_args()
    resize_images(args.input_dir, args.output_dir)
