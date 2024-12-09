import os
import yaml
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import multiprocessing
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from ultralytics import YOLO

class DataPreprocessor:
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.dataset_dir = Path('dataset')
        self.images_dir = self.dataset_dir / 'images'
        self.labels_dir = self.dataset_dir / 'labels'
        
        # Create directories for train/val/test splits
        self.splits = ['train', 'val', 'test']
        self.image_dirs = {split: self.images_dir / split for split in self.splits}
        self.label_dirs = {split: self.labels_dir / split for split in self.splits}

    def setup_directories(self):
        """Create necessary directories for the dataset."""
        for dir_path in list(self.image_dirs.values()) + list(self.label_dirs.values()):
            if dir_path.exists():
                shutil.rmtree(dir_path)
            dir_path.mkdir(parents=True, exist_ok=True)

    def load_data(self):
        """Load and prepare training and test data."""
        self.train_df = pd.read_csv(self.data_dir / 'Train.csv')
        self.test_df = pd.read_csv(self.data_dir / 'Test.csv')
        
        # Add image paths
        self.train_df['image_path'] = [Path('images/' + x) for x in self.train_df.Image_ID]
        self.test_df['image_path'] = [Path('images/' + x) for x in self.test_df.Image_ID]
        
        # Create class mapping
        classes = sorted(self.train_df['class'].unique().tolist())
        self.class_mapper = {x:y for x,y in zip(classes, range(len(classes)))}
        self.train_df['class_id'] = self.train_df['class'].map(self.class_mapper)
        
        return classes

    def split_data(self, test_size=0.25, random_state=42):
        """Split data into training and validation sets."""
        train_unique = self.train_df.drop_duplicates(subset=['Image_ID'], ignore_index=True)
        train_imgs, val_imgs = train_test_split(
            train_unique, 
            test_size=test_size,
            stratify=train_unique['class'],
            random_state=random_state
        )
        
        self.train_set = self.train_df[self.train_df.Image_ID.isin(train_imgs.Image_ID)]
        self.val_set = self.train_df[self.train_df.Image_ID.isin(val_imgs.Image_ID)]
        
        return self.train_set, self.val_set

    def copy_images(self):
        """Copy images to their respective directories."""
        # Copy training images
        for img in tqdm(self.train_set.image_path.unique(), desc="Copying training images"):
            shutil.copy(img, self.image_dirs['train'] / img.parts[-1])
        
        # Copy validation images
        for img in tqdm(self.val_set.image_path.unique(), desc="Copying validation images"):
            shutil.copy(img, self.image_dirs['val'] / img.parts[-1])
        
        # Copy test images
        for img in tqdm(self.test_df.image_path.unique(), desc="Copying test images"):
            shutil.copy(img, self.image_dirs['test'] / img.parts[-1])

    @staticmethod
    def convert_to_yolo(bbox, width, height):
        """Convert bounding box coordinates to YOLO format."""
        ymin, xmin, ymax, xmax = bbox['ymin'], bbox['xmin'], bbox['ymax'], bbox['xmax']
        class_id = bbox['class_id']
        
        x_center = (xmin + xmax) / 2 / width
        y_center = (ymin + ymax) / 2 / height
        bbox_width = (xmax - xmin) / width
        bbox_height = (ymax - ymin) / height
        
        return f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}"

    def process_single_image(self, task):
        """Process a single image and save its annotations."""
        image_path, bboxes, output_dir = task
        try:
            img = np.array(Image.open(str(image_path)))
            height, width, _ = img.shape
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return
        
        label_file = Path(output_dir) / f"{Path(image_path).stem}.txt"
        with open(label_file, 'w') as f:
            for bbox in bboxes:
                annotation = self.convert_to_yolo(bbox, width, height)
                f.write(f"{annotation}\n")

    def create_labels(self, dataset, output_dir):
        """Create YOLO format labels for a dataset."""
        if Path(output_dir).exists():
            shutil.rmtree(output_dir)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        grouped = dataset.groupby('image_path')
        tasks = [(image_path, group.to_dict('records'), output_dir) 
                for image_path, group in grouped]
        
        with multiprocessing.Pool() as pool:
            list(tqdm(
                pool.imap_unordered(self.process_single_image, tasks),
                total=len(tasks),
                desc="Creating labels"
            ))

    def create_yaml(self, class_names):
        """Create YAML configuration file for YOLO."""
        data_yaml = {
            'train': str(self.image_dirs['train']),
            'val': str(self.image_dirs['val']),
            'test': str(self.image_dirs['test']),
            'nc': len(class_names),
            'names': class_names
        }
        
        with open('data.yaml', 'w') as file:
            yaml.dump(data_yaml, file, default_flow_style=False)

class ModelTrainer:
    def __init__(self, model_name='yolo11x.pt'):
        self.model = YOLO(model_name)
    
    def train(self, epochs=100, imgsz=1024, batch=8, patience=5):
        """Train the model."""
        self.model.train(
            data='data.yaml',
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=0,
            patience=patience
        )
    
    def validate(self):
        """Validate the model."""
        return self.model.val()

    def predict(self, test_dir):
        """Make predictions on test images."""
        image_files = os.listdir(test_dir)
        all_predictions = []
        
        for image_file in tqdm(image_files, desc="Making predictions"):
            img_path = os.path.join(test_dir, image_file)
            results = self.model(img_path, stream=False)
            
            boxes = results[0].boxes.xyxy.tolist()
            classes = results[0].boxes.cls.tolist()
            confidences = results[0].boxes.conf.tolist()
            names = results[0].names
            
            for box, cls, conf in zip(boxes, classes, confidences):
                x1, y1, x2, y2 = box
                detected_class = names[int(cls)]
                
                all_predictions.append({
                    'Image_ID': image_file,
                    'class': detected_class,
                    'confidence': conf,
                    'ymin': y1,
                    'xmin': x1,
                    'ymax': y2,
                    'xmax': x2
                })
        
        return pd.DataFrame(all_predictions)

def main():
    # Initialize preprocessor
    data_dir = Path('')
    preprocessor = DataPreprocessor(data_dir)
    
    # Setup and preprocess data
    preprocessor.setup_directories()
    class_names = preprocessor.load_data()
    train_set, val_set = preprocessor.split_data()
    
    # Copy images and create labels
    preprocessor.copy_images()
    preprocessor.create_labels(train_set, preprocessor.label_dirs['train'])
    preprocessor.create_labels(val_set, preprocessor.label_dirs['val'])
    
    # Create YAML configuration
    preprocessor.create_yaml(class_names)
    
    # Train model
    trainer = ModelTrainer()
    trainer.train()
    
    # Validate model
    trainer.validate()
    
    # Make predictions on test set
    predictions = trainer.predict('dataset/images/test')
    predictions.to_csv('submission.csv', index=False)

if __name__ == "__main__":
    main()
