import cv2
import numpy as np

# Camera intrinsic parameters
focal_length = 800  # Example value
center_x = 640  # Example value for a 1280x720 image
center_y = 360  # Example value for a 1280x720 image
camera_matrix = np.array([
    [focal_length, 0, center_x],
    [0, focal_length, center_y],
    [0, 0, 1]
], dtype=np.float32)
dist_coeffs = np.zeros((4, 1), dtype=np.float32)  # Assuming no distortion

# Define 3D model points of the object
object_points = np.array([
    [0, 0, 0],  # Point 1
    [0, 1, 0],  # Point 2
    [1, 1, 0],  # Point 3
    [1, 0, 0],  # Point 4
], dtype=np.float32)

# Initialize video capture
cap = cv2.VideoCapture('/Users/janpoonthong/Downloads/hello.mp4')  # Replace with the path to your video or use 0 for webcam

# Initialize ORB detector
orb = cv2.ORB_create()

# Create a feature matcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Initialize variables
prev_frame_gray = None
prev_keypoints = None
prev_descriptors = None
prev_homography = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = orb.detectAndCompute(gray, None)

    if prev_frame_gray is not None and prev_keypoints is not None and prev_descriptors is not None:
        # Match descriptors from previous frame to current frame
        matches = bf.match(prev_descriptors, descriptors)
        matches = sorted(matches, key=lambda x: x.distance)

        # Extract matched points
        src_pts = np.float32([prev_keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Compute homography matrix
        if len(src_pts) >= 4:
            homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
            if homography is not None:
                prev_homography = homography
                h, w = frame.shape[:2]

                # Use the homography to transform the object points into the image plane
                object_points_2d = np.array([
                    [0, 0],
                    [1, 0],
                    [1, 1],
                    [0, 1]
                ], dtype=np.float32).reshape(-1, 1, 2)

                # Compute the projected 2D points
                object_points_proj = cv2.perspectiveTransform(object_points_2d.reshape(-1, 1, 2), homography)

                # Solve PnP to get rotation and translation vectors
                success, rotation_vector, translation_vector = cv2.solvePnP(object_points, object_points_proj.reshape(-1, 2), camera_matrix, dist_coeffs)

                if success:
                    # Project 3D points onto the 2D image
                    axis_length = 1.0
                    axis_points = np.array([
                        [0, 0, 0],
                        [axis_length, 0, 0],
                        [0, axis_length, 0],
                        [0, 0, axis_length]
                    ], dtype=np.float32)
                    
                    img_points, _ = cv2.projectPoints(axis_points, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
                    img_points = np.int32(img_points).reshape(-1, 2)

                    # Draw axis lines
                    cv2.line(frame, tuple(img_points[0]), tuple(img_points[1]), (0, 0, 255), 3)  # X axis in red
                    cv2.line(frame, tuple(img_points[0]), tuple(img_points[2]), (0, 255, 0), 3)  # Y axis in green
                    cv2.line(frame, tuple(img_points[0]), tuple(img_points[3]), (255, 0, 0), 3)  # Z axis in blue

    # Display the result
    cv2.imshow('Pose Estimation', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Update previous frame variables
    prev_frame_gray = gray
    prev_keypoints = keypoints
    prev_descriptors = descriptors

cap.release()
cv2.destroyAllWindows()
