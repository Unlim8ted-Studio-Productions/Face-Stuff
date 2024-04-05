import cv2
import numpy as np
import trimesh
import pyrender
from mediapipe.python.solutions import face_mesh, drawing_utils, hands, pose
from scipy.spatial import Delaunay
import trimesh.transformations as tf
import bpy
class land():
   def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
# Initialize Mediapipe Face Mesh
mp_pose_mesh = pose
mp_face_mesh = face_mesh
mp_hands_mesh = hands
maxpeople=1
pose_mesh=mp_pose_mesh.Pose()
face_meshh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=maxpeople, min_detection_confidence=0.5)
hand_mesh = mp_hands_mesh.Hands(False, maxpeople*2, min_detection_confidence=.5)
camera_pose=np.eye(4)
scale=1
face_data=[] #list of face mesh pyrender.Mesh objects, ex. [pyrender.Mesh, pyrender.Mesh]
poseandhand_data = [] #list of lists, each containing pyrender.Mesh object, each mesh will be used as a bone ex. [[pyrender.Mesh, pyrender.Mesh], [pyrender.Mesh, pyrender.Mesh]]
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
def clean_mesh(trimesh_mesh: trimesh.Trimesh):
    # Remove duplicate vertices
    #trimesh_mesh.remove_duplicate_vertices()

    # Remove degenerate faces
    trimesh_mesh.remove_degenerate_faces()

    # Remove duplicate faces
    trimesh_mesh.update_faces(trimesh_mesh.unique_faces())

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

def combine_meshes(meshes):
    """
    Combines multiple mesh objects into a single mesh.

    Args:
        trimesh.Trimesh (list): List of mesh objects.

    Returns:
        pyrender.Mesh: Combined mesh object.
    """
    # Concatenate vertices, normals, and faces
    vertices = []
   # normals = []
    faces = []

    for mesh in meshes:
        vertices.append(mesh.vertices)
        #normals.append(mesh.normals)
        faces.append(mesh.faces + len(vertices) - 1)

    # Convert lists to numpy arrays
    vertices = np.concatenate(vertices)
    #normals = np.concatenate(normals)
    faces = np.concatenate(faces)
    # Create a new Mesh object
    combined_mesh = pyrender.Mesh.from_points(vertices)#, normals = normals, faces=faces)
    return combined_mesh

def create_mesh_between_points(point1, point2, radius=0.03, num_segments=8):
    """
    Creates a mesh that extends between two points.

    Args:
        point1 (array-like): Coordinates of the first point.
        point2 (array-like): Coordinates of the second point.
        radius (float): Radius of the cylinder.
        num_segments (int): Number of segments to use for the cylinder.

    Returns:
        pyrender.Mesh: Mesh object representing the cylinder.
    """
    # Calculate direction vector and length
    direction = np.array(point2) - np.array(point1)
    length = np.linalg.norm(direction)  


    # Create cylinder vertices
    circle = np.linspace(0, 2 * np.pi, num_segments, endpoint=False)
    vertices = np.empty((num_segments * 2, 3))

    # Calculate rotation matrix to align cylinder with direction vector
    z_axis = np.array([0, 0, 1])
    if not np.allclose(direction, z_axis):
        axis = np.cross(z_axis, direction)
        axis /= np.linalg.norm(axis)
        angle = np.arccos(np.dot(z_axis, direction) / length)
        rotation_matrix = rotation_matrix_from_axis_angle(axis, angle)
    else:
        rotation_matrix = np.identity(3)

    def _create_ring(vertices, location):
        # Generate vertices along the cylinder axis
        for i in range(num_segments):
            angle = 2 * np.pi * i / num_segments
            rotated = np.dot(rotation_matrix, np.array([radius * np.cos(angle), radius * np.sin(angle), 0]))
            vertices[i] = location + rotated - direction / 2
            vertices[num_segments + i] = location + rotated + direction / 2
    _create_ring(vertices, point1)
    _create_ring(vertices, point2)
    tri = delaunay_max_distance(vertices[:, :2],5)  # Only consider x and y coordinates for 2D triangulation
    faces = tri.simplices
    

    return trimesh.Trimesh(vertices, faces)

def rotation_matrix_from_axis_angle(axis, angle):
    """
    Generates a rotation matrix from an axis-angle representation.

    Args:
        axis (array-like): 3D vector representing the rotation axis.
        angle (float): Angle of rotation in radians.

    Returns:
        numpy.ndarray: Rotation matrix.
    """
    c = np.cos(angle)
    s = np.sin(angle)
    t = 1 - c
    x, y, z = axis / np.linalg.norm(axis)
    return np.array([
        [t*x*x + c,    t*x*y - z*s,  t*x*z + y*s],
        [t*x*y + z*s,  t*y*y + c,    t*y*z - x*s],
        [t*x*z - y*s,  t*y*z + x*s,  t*z*z + c]
    ])

def create_head_mesh_from_landmarks(landmarks, scale=1.0, hand=False, connections=mp_pose_mesh.POSE_CONNECTIONS):
    global poseandhand_data, face_data
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
        if len(vertices) >= 4:
            tri = delaunay_max_distance(vertices[:, :2],5)  # Only consider x and y coordinates for 2D triangulation

            vertices[:, 1] *= -1


            # Extract the indices of the vertices forming the triangles
            faces = tri.simplices

            # Reverse the order of the faces to maintain the correct orientation
            faces = np.flip(faces, axis=1)

            mesh_trimesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            mesh = clean_mesh(mesh_trimesh)
        else:
            mesh = pyrender.mesh.Mesh.from_points(vertices, (255,0,0))
            face_data.append(trimesh.Trimesh(vertices))
    else:
        vertices[:, 1] *= -1
        # Create lines connecting vertices
        line_mesh = []
        for connection in connections:
            try:
                line_mesh.append(create_mesh_between_points(vertices[connection[0]],vertices[connection[1]]))
            except Exception as e:
                print(f"warning: {e}")
        mesh=[]
        for i in line_mesh:
            mesh.append(clean_mesh(i))
    return mesh

def track_face():
    global scene, head_mesh, camera_pose, scale, poseandhand_data
    cap = cv2.VideoCapture(0)
    head_renderer = create_head_renderer(head_mesh)
    
    running = True
    while running:
        ret, frame = cap.read()
        
        if not ret:
            break
        anim=[]
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
                hand_meshx = create_head_mesh_from_landmarks(hand_landmarks.landmark, scale, True, mp_hands_mesh.HAND_CONNECTIONS)
    
                thingstoadd.extend(hand_meshx)
                anim.extend(hand_meshx)
        results=pose_mesh.process(rgb_frame)
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            l = []
            for ll in landmarks:
                l.append(land(ll.x,ll.y,ll.z))
            pose_meshx = create_head_mesh_from_landmarks(l, scale, True)
                
            thingstoadd.extend(pose_meshx)
            anim.extend(pose_meshx)
        poseandhand_data.append(anim)
                
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
                scale=-.01
        elif key == ord('w'):
            scale += moveamount  # Move camera backward
            if scale == 0:
                scale = .01
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
    #print(f"poseandhand_data:\n{poseandhand_data}facialdata:\n{face_data}")
    
def create_3d_model_with_animations(face_data, poseandhand_data, output_path):
    # Create armature
    bpy.ops.object.armature_add(enter_editmode=False, location=(0, 0, 0))
    armature_obj = bpy.context.object
    armature_obj.name = "Armature"
    poseandhand_data = [o for o in poseandhand_data if o]
            
    bpy.ops.object.mode_set(mode='EDIT')
    for i, frame_data in enumerate(poseandhand_data):
        for j, mesh in enumerate(frame_data):
            bbox_min, bbox_max = mesh.bounds
            bone = armature_obj.data.edit_bones.new(f"Bone_{i}_{j}")
            bone.head = (bbox_min[0], bbox_min[1], bbox_min[2])
            bone.tail = (bbox_max[0], bbox_max[1], bbox_max[2])  # Adjust the bone length as needed

    bpy.ops.object.mode_set(mode='OBJECT')

    # Create shape keys for face mesh
    bpy.ops.mesh.primitive_cube_add(size=2, enter_editmode=False, align='WORLD', location=(0, 0, 0))
    face_mesh_obj = bpy.context.object
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.delete(type='VERT')
    bpy.ops.object.mode_set(mode='OBJECT')
    
    bpy.context.view_layer.objects.active = face_mesh_obj
    bpy.ops.object.shape_key_add(from_mix=False)
    basis_key = face_mesh_obj.data.shape_keys.key_blocks[0]
    basis_key.name = "Basis"
    
    num_frames = len(face_data)
    for i in range(num_frames):
        bpy.ops.object.shape_key_add(from_mix=False)
        frame_key = face_mesh_obj.data.shape_keys.key_blocks[i + 1]
        frame_key.name = f"Frame_{i}"
        # Apply vertex positions for each frame
        verts = face_data[i].vertices
        for v_index, co in enumerate(verts):
            face_mesh_obj.data.shape_keys.key_blocks[i + 1].data[v_index].co = co

    # Parent armature to face mesh
    bpy.context.view_layer.objects.active = face_mesh_obj
    bpy.ops.object.select_all(action='DESELECT')
    armature_obj.select_set(True)
    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.parent_set(type='OBJECT', keep_transform=True)

    # Set keyframe for each bone at each frame
    bpy.context.scene.frame_end = num_frames
    for i, frame_data in enumerate(poseandhand_data):
        bpy.context.scene.frame_set(i)
        for j, mesh in enumerate(frame_data):
            bone_name = f"Bone_{i}_{j}"
            bbox_min, bbox_max = mesh.bounds
            translation = (bbox_min[0] + bbox_max[0]) / 2, (bbox_min[1] + bbox_max[1]) / 2, (bbox_min[2] + bbox_max[2]) / 2
            armature_obj.pose.bones[bone_name].location = translation
            scale = (bbox_max[0] - bbox_min[0]) / 2, (bbox_max[1] - bbox_min[1]) / 2, (bbox_max[2] - bbox_min[2]) / 2
            armature_obj.pose.bones[bone_name].scale = scale
            armature_obj.pose.bones[bone_name].keyframe_insert(data_path="location")
            armature_obj.pose.bones[bone_name].keyframe_insert(data_path="scale")
            

    # Export to file
    bpy.ops.export_scene.fbx(filepath=output_path, use_selection=True)


track_face()
#create_3d_model_with_animations(face_data, poseandhand_data, "full body tracking.fbx")