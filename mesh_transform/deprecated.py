# import pickle
#
# import cv2
# from PIL import Image
#
#
# with open("C:/Users/parsa/Documents/Robot_Vision/scene1-10/scene01/labels/000000_label.pkl", 'rb') as f:
#     data = pickle.load(f)
#
# path = r'C:/Users/parsa/Documents/Robot_Vision/scene1-10/scene01/rgb/000000.png'
#
# image = cv2.imread(path)
# window_name = 'Image'
# start_point = (463, 215)
#
# end_point = (633, 416)
#
# # Blue color in BGR
# color = (255, 0, 0)
#
# # Line thickness of 2 px
# thickness = 2
#
# # Using cv2.rectangle() method
# # Draw a rectangle with blue line borders of thickness of 2 px
# image = cv2.rectangle(image, start_point, end_point, color, thickness)
#
# # Displaying the image
# cv2.imshow(window_name, image)
# cv2.waitKey()
#