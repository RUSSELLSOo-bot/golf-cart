#import opencv
import cv2 
#import matplotlib so we can visualize the image
import matplotlib.pyplot as plt


# cap = cv2.VideoCapture(0) # 0 is the default camera


# ret, frame = cap.read() # read the camera frame

# print(ret)
# print(frame)

# plt.imshow(frame[..., ::-1])  # convert BGR→RGB so colors look right
# plt.axis('off')               # (optional) hide axes
# plt.show()  

# cap.release()

def render():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
    
        # Display the resulting frame
        cv2.imshow('Camera Feed', frame)

        #press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

render()
    

