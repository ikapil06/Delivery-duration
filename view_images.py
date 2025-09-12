"""
Display the saved prediction images
"""

import os
import matplotlib.pyplot as plt
from PIL import Image

def view_saved_images():
    """Display the saved prediction images"""
    
    # Find the latest report directory
    outputs_dir = "outputs"
    report_dirs = [d for d in os.listdir(outputs_dir) if d.startswith("sample_report_")]
    if not report_dirs:
        print("No sample reports found!")
        return
    
    latest_dir = sorted(report_dirs)[-1]
    report_path = os.path.join(outputs_dir, latest_dir)
    
    # Find image files
    image_files = [f for f in os.listdir(report_path) if f.endswith('.png')]
    
    if not image_files:
        print("No images found!")
        return
    
    print(f"Found {len(image_files)} images in {latest_dir}:")
    for img in image_files:
        print(f"  - {img}")
    
    # Display images
    fig, axes = plt.subplots(1, len(image_files), figsize=(15, 6))
    if len(image_files) == 1:
        axes = [axes]
    
    for i, img_file in enumerate(image_files):
        img_path = os.path.join(report_path, img_file)
        img = Image.open(img_path)
        axes[i].imshow(img)
        axes[i].set_title(img_file.replace('.png', '').replace('_', ' ').title())
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nImages displayed from: {report_path}")

if __name__ == "__main__":
    view_saved_images()
