import cv2
import numpy as np
import trimesh
import pyrender
from mediapipe.python.solutions import face_mesh, drawing_utils
from scipy.spatial import cKDTree

# Initialize Mediapipe Face Mesh
mp_face_mesh = face_mesh
face_meshh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=5, min_detection_confidence=0.5)

# Load 3D head mesh
head_mesh = trimesh.load("Head.obj")  
scene = pyrender.Scene()
# Function to create a renderer for the 3D head
def create_head_renderer(mesh):
    global scene
    #scene.add(pyrender.Mesh.from_trimesh(mesh))
    light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=5.0)
    scene.add(light, pose=np.eye(4))
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
    camera_pose = np.eye(4)
    camera_pose[:3, :3] = np.eye(3)
    camera_pose[2, 3] = 2
    scene.add(camera, pose=camera_pose)
    return pyrender.OffscreenRenderer(viewport_width=640, viewport_height=480)

def create_head_mesh_from_landmarks(landmarks):

    # Extract coordinates of face landmarks
    vertices = np.array([(lm.x, lm.y, lm.z) for lm in landmarks])
    
    # Create mesh from landmarks and triangles
    mesh = pyrender.mesh.Mesh.from_points(vertices)
    
    return mesh

def track_face():
    global scene, head_mesh
    cap = cv2.VideoCapture(0)
    head_renderer = create_head_renderer(head_mesh)
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Convert the frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect face landmarks
        results = face_meshh.process(rgb_frame)  # Assuming face_mesh is initialized somewhere
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Create head mesh from face landmarks
                head_meshx = create_head_mesh_from_landmarks(face_landmarks.landmark)
                
                # Render the 3D head
                scene.mesh_nodes.clear()
                scene.add(head_meshx)
                color, _ = head_renderer.render(scene)
                head_image = color.copy()
                head_image = cv2.cvtColor(head_image, cv2.COLOR_RGBA2BGR)
                
                # Display the 3D head
                cv2.imshow('3D Head', head_image)
        
        # Display the frame with facial landmarks
        cv2.imshow('Face Tracking', frame)
        
        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
track_face()
