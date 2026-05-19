import cv2
import face_recognition
import numpy as np
import os
import json
from datetime import datetime

class FacialRecognitionSystem:
    def __init__(self, database_path="known_faces_db.json"):
        # Initialize database for known faces
        self.database_path = database_path
        self.known_face_encodings = []
        self.known_face_names = []
        self.known_face_relations = []
        
        # Load database if exists
        self.load_database()
        
        # Initialize camera
        self.video_capture = None
        
    def load_database(self):
        """Load known faces from database file if it exists"""
        if os.path.exists(self.database_path):
            try:
                with open(self.database_path, 'r') as f:
                    data = json.load(f)
                    
                for person in data:
                    self.known_face_encodings.append(np.array(person["encoding"]))
                    self.known_face_names.append(person["name"])
                    self.known_face_relations.append(person["relation"])
                
                print(f"Loaded {len(self.known_face_names)} faces from database")
            except Exception as e:
                print(f"Error loading database: {e}")
                # Create empty database if loading fails
                self.create_empty_database()
        else:
            # Create empty database if not exists
            self.create_empty_database()
    
    def create_empty_database(self):
        """Create an empty database file"""
        with open(self.database_path, 'w') as f:
            json.dump([], f)
        print("Created new empty database")
    
    def save_to_database(self):
        """Save current known faces to database file"""
        data = []
        for i in range(len(self.known_face_names)):
            person = {
                "name": self.known_face_names[i],
                "relation": self.known_face_relations[i],
                "encoding": self.known_face_encodings[i].tolist()
            }
            data.append(person)
        
        with open(self.database_path, 'w') as f:
            json.dump(data, f)
        print("Database updated")
    
    def add_new_face(self, frame):
        """Add a new face to the database"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        
        if len(face_locations) == 0:
            print("No face detected!")
            return False
        
        if len(face_locations) > 1:
            print("Multiple faces detected. Please ensure only one person is in frame.")
            return False
        
        face_encoding = face_recognition.face_encodings(rgb_frame, face_locations)[0]
        
        # Ask for person's name and relation
        name = input("Enter person's name: ")
        relation = input("Enter relation to you: ")
        
        # Add to known faces
        self.known_face_encodings.append(face_encoding)
        self.known_face_names.append(name)
        self.known_face_relations.append(relation)
        
        # Save to database
        self.save_to_database()
        print(f"Added {name} to database!")
        return True
    
    def start_camera(self):
        """Start the camera and facial recognition process"""
        self.video_capture = cv2.VideoCapture(0)
        
        if not self.video_capture.isOpened():
            print("Error: Could not open camera.")
            return
        
        process_this_frame = True
        
        while True:
            # Grab a single frame
            ret, frame = self.video_capture.read()
            if not ret:
                print("Failed to grab frame")
                break
                
            # Only process every other frame to save time
            if process_this_frame:
                # Resize frame for faster processing
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                
                # Convert from BGR (OpenCV) to RGB (face_recognition)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                
                # Find all faces in the frame
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                
                face_names = []
                face_relations = []
                
                for face_encoding in face_encodings:
                    # See if the face matches any known face
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                    name = "Unknown"
                    relation = ""
                    
                    # Use the known face with the smallest distance to the new face
                    if len(self.known_face_encodings) > 0:
                        face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                        best_match_index = np.argmin(face_distances)
                        if matches[best_match_index]:
                            name = self.known_face_names[best_match_index]
                            relation = self.known_face_relations[best_match_index]
                    
                    face_names.append(name)
                    face_relations.append(relation)
            
            process_this_frame = not process_this_frame
            
            # Display the results
            for (top, right, bottom, left), name, relation in zip(face_locations, face_names, face_relations):
                # Scale back up locations since we resized the frame
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                
                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                
                # Draw a label with the name below the face
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                
                if name == "Unknown":
                    label_text = "Unknown Person"
                else:
                    label_text = f"{name} - {relation}"
                
                cv2.putText(frame, label_text, (left + 6, bottom - 6), font, 0.6, (255, 255, 255), 1)
            
            # Display help text
            cv2.putText(frame, "Press 'a' to add new face, 'q' to quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display the resulting image
            cv2.imshow('Face Recognition System', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                # Quit
                break
            elif key == ord('a'):
                # Add new face
                print("Capturing image for new face registration...")
                # Give a little pause to prepare
                for i in range(3, 0, -1):
                    print(f"Capturing in {i}...")
                    cv2.waitKey(1000)
                
                # Capture a clean frame
                ret, add_frame = self.video_capture.read()
                if ret:
                    self.add_new_face(add_frame)
        
        # Release handle to the camera
        self.video_capture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Starting Facial Recognition System")
    print("-----------------------------------")
    system = FacialRecognitionSystem()
    
    if len(system.known_face_names) == 0:
        print("No known faces in database. You'll need to add some!")
    else:
        print(f"Database contains {len(system.known_face_names)} known faces")
    
    print("Opening camera...")
    system.start_camera()