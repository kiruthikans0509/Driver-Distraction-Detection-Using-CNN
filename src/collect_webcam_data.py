import cv2
import os
from datetime import datetime

# Same class mapping as test_facemesh.py
DISTRACTION_CLASSES = [
    "Safe driving",           # c0
    "Texting - right",        # c1
    "Talking on phone - right",  # c2
    "Texting - left",         # c3
    "Talking on phone - left",   # c4
    "Operating radio",        # c5
    "Drinking",                # c6
    "Reaching behind",        # c7
    "Hair/makeup",             # c8
    "Talking to passenger"    # c9
]

OUTPUT_DIR = "data/webcam_distraction"
SESSION = "session3_raw"  # Raw collection target for Session 3

def create_directories():
    """Create output directory and class folders if they don't exist"""
    session_dir = os.path.join(OUTPUT_DIR, SESSION)
    if not os.path.exists(session_dir):
        os.makedirs(session_dir)
    
    for i in range(10):
        class_dir = os.path.join(session_dir, f"c{i}")
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)

def get_image_count(class_idx):
    """Get the number of images already collected for a class"""
    session_dir = os.path.join(OUTPUT_DIR, SESSION)
    class_dir = os.path.join(session_dir, f"c{class_idx}")
    if os.path.exists(class_dir):
        return len([f for f in os.listdir(class_dir) if f.endswith('.jpg')])
    return 0

def get_next_image_number(class_idx):
    """Get the next available image number for a class"""
    return get_image_count(class_idx) + 1

def main():
    create_directories()
    
    print("=== Webcam Data Collection Tool ===")
    print(f"Current session: {SESSION}")
    print(f"Output directory: {os.path.join(OUTPUT_DIR, SESSION)}")
    print("Controls:")
    print("  SPACE = Capture current frame")
    print("  N = Next class")
    print("  P = Previous class")
    print("  Q/ESC = Quit")
    print()
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    current_class_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Get current class info
        class_name = DISTRACTION_CLASSES[current_class_idx]
        image_count = get_image_count(current_class_idx)
        
        # Create overlay frame
        overlay = frame.copy()
        
        # Display current class info
        cv2.putText(overlay, f"Session: {SESSION}", 
                    (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(overlay, f"Class: c{current_class_idx} - {class_name}", 
                    (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(overlay, f"Images collected: {image_count}", 
                    (30, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display instructions
        cv2.putText(overlay, "SPACE = capture  N = next  P = prev  Q/ESC = quit", 
                    (30, overlay.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Display the frame
        cv2.imshow("Webcam Data Collection", overlay)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' ') or key == ord(' '):  # SPACE to capture
            # Generate filename
            img_number = get_next_image_number(current_class_idx)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"img_{img_number:04d}_{timestamp}.jpg"
            session_dir = os.path.join(OUTPUT_DIR, SESSION)
            filepath = os.path.join(session_dir, f"c{current_class_idx}", filename)
            
            # Ensure filename doesn't already exist
            while os.path.exists(filepath):
                img_number += 1
                filename = f"img_{img_number:04d}_{timestamp}.jpg"
                filepath = os.path.join(session_dir, f"c{current_class_idx}", filename)
            
            # Save the frame (original resolution, no preprocessing)
            cv2.imwrite(filepath, frame)
            print(f"Saved: {filename} to {SESSION}/c{current_class_idx} ({class_name})")
            
        elif key == ord('n') or key == ord('N'):  # Next class
            current_class_idx = (current_class_idx + 1) % 10
            print(f"Moved to class c{current_class_idx}: {DISTRACTION_CLASSES[current_class_idx]}")
            
        elif key == ord('p') or key == ord('P'):  # Previous class
            current_class_idx = (current_class_idx - 1) % 10
            print(f"Moved to class c{current_class_idx}: {DISTRACTION_CLASSES[current_class_idx]}")
            
        elif key == ord('q') or key == ord('Q') or key == 27:  # Q or ESC to quit
            print("Quitting...")
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    # Print summary
    session_dir = os.path.join(OUTPUT_DIR, SESSION)
    print("\n=== Collection Summary ===")
    for i in range(10):
        count = get_image_count(i)
        print(f"c{i} ({DISTRACTION_CLASSES[i]}): {count} images")
    print(f"\nTotal images: {sum(get_image_count(i) for i in range(10))}")
    print(f"Saved in: {session_dir}/")

if __name__ == "__main__":
    main()