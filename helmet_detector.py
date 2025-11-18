import cv2
import os
import sys
from pathlib import Path
from ultralytics import YOLO
import argparse


class HelmetDetector:
    def __init__(self, model_path='yolov8n.pt', conf_threshold=0.25, save_output=True):
        """
        Initialize the Helmet Detector
        
        Args:
            model_path: Path to YOLO model file (use yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt)
            conf_threshold: Confidence threshold for detections
            save_output: Whether to save the output video
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.save_output = save_output
        self.model = None
        
    def load_model(self):
        """Load the YOLO model"""
        try:
            print(f"Loading model: {self.model_path}")
            
            # Check if this is a HuggingFace model
            if '/' in self.model_path and not os.path.exists(self.model_path):
                print("Detected HuggingFace model, downloading...")
                try:
                    # Try loading directly with YOLO (supports HF models)
                    self.model = YOLO(self.model_path)
                    print("Model loaded successfully from HuggingFace!")
                except Exception as e1:
                    print(f"Direct loading failed: {e1}")
                    print("Attempting to download using huggingface_hub...")
                    try:
                        from huggingface_hub import hf_hub_download
                        print("Downloading model from HuggingFace...")
                        model_file = hf_hub_download(repo_id=self.model_path, filename="best.pt", cache_dir=".")
                        self.model = YOLO(model_file)
                        print("Model downloaded and loaded successfully!")
                    except ImportError:
                        print("Installing huggingface_hub...")
                        os.system("pip install huggingface_hub -q")
                        from huggingface_hub import hf_hub_download
                        print("Downloading model from HuggingFace...")
                        model_file = hf_hub_download(repo_id=self.model_path, filename="best.pt", cache_dir=".")
                        self.model = YOLO(model_file)
                        print("Model downloaded and loaded successfully!")
                    except Exception as e2:
                        print(f"Could not load from HuggingFace: {e2}")
                        raise e2
            else:
                self.model = YOLO(self.model_path)
                print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            print(f"\nTrying alternative loading method...")
            try:
                # Try YOLO's built-in HF support
                print("Attempting direct YOLO loading...")
                self.model = YOLO(self.model_path)
                print("Success!")
            except:
                print("\nPlease check:")
                print("1. Internet connection for downloading models")
                print("2. Model path is correct")
                print("3. Or use a local model file path")
            sys.exit(1)
    
    def detect_helmets(self, video_path, output_path=None):
        """
        Detect helmets in a video
        
        Args:
            video_path: Path to input video file
            output_path: Path to save output video (optional)
        """
        if not os.path.exists(video_path):
            print(f"Error: Video file not found: {video_path}")
            return
        
        if self.model is None:
            self.load_model()
        
        # Generate output path if not provided
        if output_path is None:
            base_name = Path(video_path).stem
            output_path = f"{base_name}_helmet_detected.mp4"
        
        print(f"\nProcessing video: {video_path}")
        print(f"Output will be saved to: {output_path}")
        
        # Open video capture
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print("Error: Could not open video file")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # Define codec and create VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = None
        if self.save_output:
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        helmet_count = 0
        
        print("\nProcessing frames...")
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Run YOLO detection
            results = self.model(frame, conf=self.conf_threshold, verbose=False)
            
            # Get all detections
            boxes = results[0].boxes
            
            # Filter to show only helmet-related detections
            helmet_detections = []
            current_helmets = 0
            
            # Process each detection
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    try:
                        cls = int(box.cls[0])
                        class_name = self.model.names[cls]
                        conf = float(box.conf[0])
                        
                        # Check if this is a helmet-related detection
                        is_helmet = False
                        # Check for safety helmet model classes
                        if ('helmet' in class_name.lower() or 
                            'safety' in class_name.lower() or
                            'hard hat' in class_name.lower() or
                            'hardhat' in class_name.lower() or
                            class_name.lower() == 'head' or
                            'headgear' in class_name.lower() or
                            'with helmet' in class_name.lower() or
                            'without helmet' in class_name.lower()):
                            is_helmet = True
                        
                        if is_helmet:
                            current_helmets += 1
                            helmet_detections.append({
                                'box': box,
                                'name': class_name,
                                'conf': conf
                            })
                    except Exception as e:
                        continue  # Skip this detection if there's an error
            
            # If no helmet-specific detections, show all detections with special highlighting for relevant ones
            annotated_frame = frame.copy()
            
            if len(helmet_detections) > 0:
                # Draw helmet detections
                for det in helmet_detections:
                    box = det['box']
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = det['conf']
                    label = f"{det['name']} {conf:.2f}"
                    
                    # Draw bounding box
                    cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
                    
                    # Draw label
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    cv2.rectangle(annotated_frame, (int(x1), int(y1) - label_size[1] - 10), 
                                 (int(x1) + label_size[0], int(y1)), (0, 255, 0), -1)
                    cv2.putText(annotated_frame, label, (int(x1), int(y1) - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            else:
                # Show all detections if no helmet-specific classes found
                # This helps identify what standard YOLO is detecting
                if boxes is not None and len(boxes) > 0:
                    if frame_count == 1:  # Only print once
                        print("\n⚠️  No helmet-specific classes detected. Showing all detections for debugging.")
                    
                    for box in boxes:
                        try:
                            cls = int(box.cls[0])
                            class_name = self.model.names[cls]
                            conf = float(box.conf[0])
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            
                            # Use different colors for different types
                            # Orange for people/vehicles (relevant to helmet context)
                            # Red for other objects
                            if class_name.lower() in ['person', 'motorcycle', 'bicycle', 'bike']:
                                color = (255, 165, 0)  # Orange for relevant
                                thickness = 2
                            else:
                                color = (0, 0, 255)  # Red for others
                                thickness = 1
                            
                            label = f"{class_name} {conf:.2f}"
                            
                            # Draw bounding box
                            cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
                            
                            # Draw label
                            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                            cv2.rectangle(annotated_frame, (int(x1), int(y1) - label_size[1] - 5), 
                                         (int(x1) + label_size[0], int(y1)), color, -1)
                            cv2.putText(annotated_frame, label, (int(x1), int(y1) - 3), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        except Exception as e:
                            continue  # Skip this detection if there's an error
            
            if current_helmets > 0:
                helmet_count += current_helmets
            
            # Write frame to output video
            if self.save_output and out is not None:
                out.write(annotated_frame)
            
            frame_count += 1
            
            # Progress update
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)")
        
        # Release resources
        cap.release()
        if out is not None:
            out.release()
        
        print(f"\n✓ Processing complete!")
        print(f"  Total frames processed: {frame_count}")
        print(f"  Total helmet detections: {helmet_count}")
        if self.save_output:
            print(f"  Output saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Helmet Detection in Video using YOLO')
    parser.add_argument('--video', type=str, required=True, help='Path to input video file')
    parser.add_argument('--model', type=str, default='yolov8n.pt', 
                       help='YOLO model to use (default: yolov8n.pt)')
    parser.add_argument('--output', type=str, default=None, 
                       help='Output video path (default: auto-generated)')
    parser.add_argument('--conf', type=float, default=0.25, 
                       help='Confidence threshold (default: 0.25)')
    parser.add_argument('--no-save', action='store_true', 
                       help='Do not save output video')
    
    args = parser.parse_args()
    
    # Create detector instance
    detector = HelmetDetector(
        model_path=args.model,
        conf_threshold=args.conf,
        save_output=not args.no_save
    )
    
    # Run detection
    detector.detect_helmets(args.video, args.output)


if __name__ == "__main__":
    main()
