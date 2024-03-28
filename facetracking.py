import cv2
import numpy as np
import trimesh
import pyrender
from mediapipe.python.solutions import face_mesh, drawing_utils, hands, pose
from scipy.spatial import Delaunay
import trimesh.transformations as tf
class land():
   def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
# Initialize Mediapipe Face Mesh
mp_pose_mesh = pose
mp_face_mesh = face_mesh
mp_hands_mesh = hands
maxpeople=5
pose_mesh=mp_pose_mesh.Pose()
face_meshh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=maxpeople, min_detection_confidence=0.5)
hand_mesh = mp_hands_mesh.Hands(False, maxpeople*2, min_detection_confidence=.5)
camera_pose=np.eye(4)
scale=1
# Load 3D head mesh
head_mesh = trimesh.load("Head.obj")  
scene = pyrender.Scene()
# Function to create a renderer for the 3D head
def create_head_renderer(mesh):
    global scene, camera_pose
    #scene.add(pyrender.Mesh.from_trimesh(mesh))
    light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=5.0)
    scene.add(light, pose=np.eye(4))
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
    camera_pose = np.eye(4)
    camera_pose[:3, :3] = np.eye(3)
    camera_pose[2, 3] = 2
    
    
#    camera_pose = np.array([
#    [ 1.00,  0.00, -0.00,  0.52],
#    [ 0.00,  1.00, -0.00,  0.63],
#    [ 0.00,  0.00, -1.00, -1.45],
#    [ 0.00,  0.00,  0.00,  1.00]
#])

    scene.add(camera, pose=camera_pose)
    renderer = pyrender.OffscreenRenderer(viewport_width=640, viewport_height=480)
    renderer.point_size = 10  # Adjust the point size here
    return renderer

def interpolate_vertices(vertices, num_interpolated_points=10):
    interpolated_vertices = []
    for i in range(len(vertices) - 1):
        start_vertex = vertices[i]
        end_vertex = vertices[i + 1]

        # Perform linear interpolation between start and end vertices
        interpolated_points = np.linspace(start_vertex, end_vertex, num_interpolated_points, axis=0)

        # Exclude the last point to avoid duplication
        interpolated_vertices.extend(interpolated_points[:-1])

    # Add the last vertex
    interpolated_vertices.append(vertices[-1])

    return np.array(interpolated_vertices)
def clean_mesh(trimesh_mesh):
    # Remove duplicate vertices
    #trimesh_mesh.remove_duplicate_vertices()

    # Remove degenerate faces
    trimesh_mesh.remove_degenerate_faces()

    # Remove duplicate faces
    trimesh_mesh.remove_duplicate_faces()

    # Remove zero area faces
    trimesh_mesh.remove_unreferenced_vertices()

    # Fix normals
    trimesh_mesh.fix_normals()
    
    # Smooth the mesh using Laplacian smoothing
   # trimesh_mesh = trimesh.smoothing.filter_laplacian(trimesh_mesh, iterations=5)
    # Remove spikes (outliers) by filtering based on vertex normals
    #cleaned_vertices = trimesh_mesh.vertices
    #cleaned_faces = trimesh_mesh.faces
    #cleaned_faces = np.clip(cleaned_faces, 0, len(cleaned_vertices) - 1)
    # Convert back to Pyrender Mesh
    cleaned_mesh = pyrender.mesh.Mesh.from_trimesh(trimesh_mesh)

    return cleaned_mesh
def delaunay_max_distance(vertices, max_distance):
    # Calculate pairwise distances between vertices
    distances = np.sqrt(np.sum((vertices[:, None] - vertices) ** 2, axis=-1))

    # Create a mask to filter distances exceeding the maximum distance
    mask = distances <= max_distance
    def _filter(vertices):
        # Apply the mask to create a new set of vertices
        filtered_vertices = vertices[mask.all(axis=1)]
        return filtered_vertices
    #while True:
    #    a=_filter(vertices)
    #    if len(a) >=4:
    #        # Perform Delaunay triangulation
    #        tri = Delaunay(a)
    #        return tri
    #    else:
    #        max_distance+=.1
    #        mask = distances <= max_distance
    filtered_vertices = vertices[mask.all(axis=1)]
    tri = Delaunay(filtered_vertices)
    return tri
        
def create_head_mesh_from_landmarks(landmarks, scale=1.0, hand=False):
    # Extract coordinates of face landmarks
    vertices = np.array([(lm.x, lm.y, lm.z) for lm in landmarks])
    #vertices = interpolate_vertices(vertices)
    # Calculate the centroid of the mesh
    centroid = np.mean(vertices, axis=0)

    # Translate vertices to the origin
    vertices -= centroid

    # Scale the vertices
    vertices *= scale

    # Translate vertices back to their original position
    vertices += centroid
    
    # Flip the vertices along the y-axis        
    if not hand:

        # Perform Delaunay triangulation
        tri = delaunay_max_distance(vertices[:, :2],1)  # Only consider x and y coordinates for 2D triangulation

        vertices[:, 1] *= -1


        # Extract the indices of the vertices forming the triangles
        faces = tri.simplices

        # Reverse the order of the faces to maintain the correct orientation
        faces = np.flip(faces, axis=1)

        mesh_trimesh = trimesh.Trimesh(vertices=vertices, faces=faces)

        mesh = clean_mesh(mesh_trimesh)
    else:
        vertices[:, 1] *= -1

        mesh=pyrender.mesh.Mesh.from_points(vertices, (255,0,0))
    
    return mesh



    
def track_face():
    global scene, head_mesh, camera_pose, scale
    cap = cv2.VideoCapture(0)
    head_renderer = create_head_renderer(head_mesh)
    
    running = True
    while running:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Convert the frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect face landmarks
        results = face_meshh.process(rgb_frame)  # Assuming face_mesh is initialized somewhere
        thingstoadd=[]
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Create head mesh from face landmarks
                head_meshx = create_head_mesh_from_landmarks(face_landmarks.landmark, scale)

                thingstoadd.append(head_meshx)
                
        results=hand_mesh.process(rgb_frame)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                hand_meshx = create_head_mesh_from_landmarks(hand_landmarks.landmark, scale, True)
    
                thingstoadd.append(hand_meshx)
        results=pose_mesh.process(rgb_frame)
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            l = []
            for ll in landmarks:
                l.append(land(ll.x,ll.y,ll.z))
            pose_meshx = create_head_mesh_from_landmarks(l, scale)
                
            thingstoadd.append(pose_meshx)
                
        scene.mesh_nodes.clear()
        for i in thingstoadd:
            scene.add(i)
        color, _ = head_renderer.render(scene)
        head_image = color.copy()
        head_image = cv2.cvtColor(head_image, cv2.COLOR_RGBA2BGR)
        # Display the 3D head
        cv2.imshow('3D Head', head_image)
        
        # Display the frame with facial landmarks
        cv2.imshow('Face Tracking', frame)

        # Handle keypress events for camera movement
        moveamount=1
        key = cv2.waitKey(1)
        if key == ord('s'):
            scale -= moveamount  # Move camera forward
            if scale == 0:
                scale=-1
        elif key == ord('w'):
            scale += moveamount  # Move camera backward
            if scale == 0:
                scale = 1
        #elif key == ord('a'):
        #    camera_pose[0, 3] -= moveamount  # Move camera left
        #elif key == ord('d'):
        #    camera_pose[0, 3] += moveamount  # Move camera right
        #    
        #scene.cameras.clear()
        #camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
        #scene.add(camera, pose=camera_pose)

        # Handle mouse click event for printing camera pose
        if cv2.getWindowProperty('Face Tracking', cv2.WND_PROP_VISIBLE) < 1:
            running = False
        elif cv2.getWindowProperty('Face Tracking', cv2.WND_PROP_AUTOSIZE) < 1:
            running = False

        # Print camera pose on mouse click event (left mouse button)
        if cv2.waitKey(1) & 0xFF == ord('p'):
            quit()
        
    # Release the camera and close all windows
    cap.release()
    cv2.destroyAllWindows()

track_face()