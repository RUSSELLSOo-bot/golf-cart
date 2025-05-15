import AIrender
import numpy as np
import cv2
import pygame
from pygame.locals import *  # This imports OPENGL, DOUBLEBUF, etc.
from OpenGL.GL import *
from OpenGL.GLU import *
import time
class Camera3D:
    def __init__(self):
        # Camera intrinsic parameters
        self.fx = 100
        self.fy = 100
        # Set principal point to image center (assuming 1920x1080)
        self.cx = 320  # width/2
        self.cy = 240  # height/2

    def unproject_point(self, x, y):
        # Center the coordinates around principal point and normalize
        x_norm = (x - self.cx) / self.fx
        y_norm = (y - self.cy) / self.fy
        # Return 3D point with z=1 (on image plane)
        return np.array([x_norm, -y_norm, 1.0])

class IsometricView:
    def __init__(self, width=800, height=600):
        pygame.init()
        # Fix: Use OPENGL and DOUBLEBUF instead of PYGAME_OPENGL and PYGAME_DOUBLEBUF
        self.display = pygame.display.set_mode((width, height), OPENGL | DOUBLEBUF)
        pygame.display.set_caption("Nose Position Tracker")
        
        # Setup OpenGL
        glEnable(GL_DEPTH_TEST)
        glViewport(0, 0, width, height)
        
        # Perspective setup
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, width/height, 0.1, 50.0)
        
        # Isometric view setup
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(1, 1, 3,    # Camera position
                  0, 0, 0,    # Look at center
                  0, 1, 0)    # Up vector

    def render(self, point):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Draw coordinate axes
        glBegin(GL_LINES)
        # X axis (red)
        glColor3f(1, 0, 0)
        glVertex3f(0, 0, 0)
        glVertex3f(1, 0, 0)
        # Y axis (green)
        glColor3f(0, 1, 0)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 1, 0)
        # Z axis (blue)
        glColor3f(0, 0, 1)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 0, 1)
        glEnd()

        # Draw nose position point
        glPointSize(10.0)
        glBegin(GL_POINTS)
        glColor3f(1, 1, 0)  # Yellow
        glVertex3f(point[0], point[1], point[2])
        glEnd()

        pygame.display.flip()

def main():
    camera = Camera3D()
    view = IsometricView()
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cap.set(cv2.CAP_PROP_FOCUS, 250)

    nose_filter = AIrender.create_calibrated_filter()
    running = True
    prev_time = time.time()
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_q):
                running = False

        # Get frame and detect nose
        ret, frame = cap.read()
        if ret:
            kps = AIrender.detect_pose(frame)
            if kps is not None and len(kps) > 0:
                y, x, conf = kps[0]  # Nose keypoint
                current_time = time.time()
                dt = current_time - prev_time
                prev_time = current_time 

                filtered_x, filtered_y = AIrender.filter_point(x, y,dt, nose_filter)

                if conf > 0.3:
                    h, w = frame.shape[:2]
                    # Center the image coordinates
                    img_x = filtered_x * w
                    img_y = filtered_y * h
                    
                    # Draw crosshair at center of frame
                    cv2.line(frame, (w//2-10, h//2), (w//2+10, h//2), (0,255,255), 1)
                    cv2.line(frame, (w//2, h//2-10), (w//2, h//2+10), (0,255,255), 1)
                    
                    # Unproject to 3D space
                    
                    point_3d = camera.unproject_point(img_x, img_y)
                    
                    # Print coordinates
                    print(f"\rNose position - X: {point_3d[0]:.3f}, Y: {point_3d[1]:.3f}, Z: {point_3d[2]:.3f}", end="")
                    
                    # Draw nose point on camera feed
                    cv2.circle(frame, (int(img_x), int(img_y)), 5, (0, 255, 0), -1)
                    cv2.putText(frame, f"({point_3d[0]:.2f}, {point_3d[1]:.2f}, {point_3d[2]:.2f})", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    # Render the point in 3D
                    view.render(point_3d)
            
            # Show camera feed
            cv2.imshow("Camera Feed", frame)
            
            # Check for 'q' key in OpenCV window
            if cv2.waitKey(1) & 0xFF == ord('q'):
                running = False

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()

if __name__ == "__main__":
    main()