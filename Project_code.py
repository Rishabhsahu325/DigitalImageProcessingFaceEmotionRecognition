import tkinter
import numpy 
import matplotlib.pyplot as plt
import cv2
from keras.models import load_model

from tkinter import filedialog
centre_x=0
centre_y=0


def detect_faces(cascade, test_image, scaleFactor = 1.1):
    # create a copy of the image to prevent any changes to the original one.
    image_copy = test_image.copy()

    #convert the test image to gray scale as opencv face detector expects gray images
    gray_image = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)

    # Applying the haar classifier to detect faces ,function returns coordinates  of corners in array form
    faces_rect = cascade.detectMultiScale(gray_image, scaleFactor=scaleFactor, minNeighbors=5)
    
    centre_x=(faces_rect[0][0]+faces_rect[0][2])/2
    centre_y=(faces_rect[0][1]+faces_rect[0][3])/2
    #display green rectangle over detected face
    for (x, y, w, h) in faces_rect:
        cv2.rectangle(image_copy, (x, y), (x+w, y+h), (0, 255, 0), 15)
        
    #return face with rectangle and coordinates of detected face
    return image_copy,faces_rect


def median_blur_func(image):    
    dst = cv2.medianBlur(image,5)
    return dst



def image_picker():
    #This place will hold code to pick up locally stored image file ,at least .png and .jpeg should be compatible
    #call the filepicker to pick up an image file
    root = tkinter.Tk()
    root.withdraw()
    #get image path in text form
    img_path = filedialog.askopenfilename()


    #call imread function with the image location obtained above to apply image processing on it
    img1=cv2.imread(img_path)
    #create resizable window to display the image
    cv2.namedWindow("Window1", cv2.WINDOW_NORMAL)
    cv2. setWindowProperty ('Window1', cv2. WND_PROP_FULLSCREEN, cv2. WINDOW_FULLSCREEN)
    #display image in the resizable window defined above to display unmodified image
    cv2.imshow("Window1",img1)
    
    #code to close the windows 
    cv2.waitKey()
    cv2.destroyAllWindows()
    return img1



def convertToRGB(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



def recognise_emotion(locations,image):
    #Using coordinates obtained from face detect to specify region of face to emotion detector
    x,y,w,h = locations[0]
    
    top=y
    right=x+w
    bottom=y+h
    left=x
    
    f_image = image[top:bottom, left:right]
    face_image= f_image
    

    #Load the model trained for detecting emotions of a face
    model = load_model("model_v6_23.hdf5")
    
    #resizing the image
    face_image = cv2.resize(face_image, (48,48))
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    face_image = numpy.reshape(face_image, [1, face_image.shape[0], face_image.shape[1], 1])

    #Creating a dictionary with emotions as keys and their numeric representations as values.
    emotion_dict= {'Angry': 0, 'Sad': 5, 'Neutral': 4, 'Disgust': 1, 'Surprise': 6, 'Fear': 2, 'Happy': 3}
    
    predicted_class = numpy.argmax(model.predict(face_image))
    label_map = dict((v,k) for k,v in emotion_dict.items())
    #Storing the emotion detected in predicted_label variable.
    predicted_label = label_map[predicted_class]
    return predicted_label
#put up statements to test your function here
if __name__ == "__main__":
  
    #Reading the original image, here 1 implies that image is read as color
    image_c=image_picker()
    
    #currently median blur with 5 as parameter (specified in the defined function ) works best
    median_blur_img=median_blur_func(image_c)

    #generate matplotlib subplot for result of denoising the image
    fig1,axs1 = plt.subplots()
    axs1.axis('off')
    axs1.set_title('Image after Denoising(Median filter)')
    plt.tight_layout()
    figManager = plt.get_current_fig_manager()  
    figManager.full_screen_toggle()
    fig1.canvas.toolbar.pack_forget()

    axs1.imshow(cv2.cvtColor(median_blur_img, cv2.COLOR_BGR2RGB),aspect='auto')
    
    #close displayed plot after pressing button
    plt.waitforbuttonpress(0)
    plt.close()

    #Applying contrast stretching   
    #Generating the histogram of the original image
    hist_c,bins_c = numpy.histogram(median_blur_img.flatten(),256,[0,256])
    #Generating the cumulative distribution function of the original image
    cdf_c = hist_c.cumsum()
    cdf_c_normalized = cdf_c * hist_c.max()/ cdf_c.max()

    #convert to YUV channel
    image_yuv = cv2.cvtColor(median_blur_img, cv2.COLOR_BGR2YUV)
    # Loop  over the Y channel and apply Min-Max Contrasting 
    min = numpy.min(image_yuv[:,:,0])
    max = numpy.max(image_yuv[:,:,0])

    for i in range(image_yuv.shape[0]):
        for j in range(image_yuv.shape[1]):
            image_yuv[:,:,0][i,j] = 255*(image_yuv[:,:,0][i,j]-min)/(max-min)
    # convert the YUV image back to RGB format
    image_c_cs = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2BGR)

    #Generating the histogram of the image after applying Constrast Stretching
    hist_c_cs, bins_c_cs = numpy.histogram(image_c_cs.flatten(),256,[0,256])
    
    #Generating the cumulative distribution function of the original image
    cdf_c_cs = hist_c_cs.cumsum()
    cdf_c_cs_normalized = cdf_c_cs * hist_c_cs.max()/ cdf_c_cs.max()

    #toggle interactive mode on that displays plots without explicitly commanding
    plt.ion()
    #generate matplotlib subplots
    fig, axs = plt.subplots(2, 2)  
    axs[0, 0].set_title('Original Image histogram') #histogram of image after denoising
    axs[1, 0].axis('off')
    #Plotting histogram of image before contrast stretching
    axs[0, 0].hist(image_c.flatten(),bins_c)
    #Displaying image before applying contrast stretching.
    axs[1, 0].imshow(cv2.cvtColor(median_blur_img, cv2.COLOR_BGR2RGB),aspect='auto')
    
    #Plotting histogram of image after contrast stretching
    axs[0, 1].hist(image_c_cs.flatten(),bins_c_cs)
    axs[0, 1].set_title('Histogram after Contrast Stretching')
    axs[1, 1].axis('off')
    #Displaying image after applying contrast stretching.
    axs[1, 1].imshow(cv2.cvtColor(image_c_cs, cv2.COLOR_BGR2RGB),aspect='auto')
    plt.tight_layout()
    figManager = plt.get_current_fig_manager()  
    figManager.full_screen_toggle()
    plt.waitforbuttonpress(0)
    plt.close()

    #generate matplotlib subplots 
    fig3,axs3 = plt.subplots()
    axs3.axis('off')
    axs3.set_title('Image after Contrast Stretching')
    plt.tight_layout()
    figManager = plt.get_current_fig_manager()  
    figManager.full_screen_toggle()
    fig3.canvas.toolbar.pack_forget()
    axs3.imshow(cv2.cvtColor(image_c_cs, cv2.COLOR_BGR2RGB),aspect='auto')
    plt.waitforbuttonpress(0)
    plt.close()
    
    # Displaying grayscale image
    #plt.imshow(test_image_gray, cmap='gray')
     
    haar_cascade_face = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')
   
    #call the function to detect faces
    faces,location = detect_faces(haar_cascade_face, image_c_cs)
    
    #generate matplotlib subplots
    fig4,axs4 = plt.subplots()
    axs4.axis('off')
    axs4.set_title('Face Detected')
    plt.tight_layout()
    figManager = plt.get_current_fig_manager()  
    figManager.full_screen_toggle()
    #For removing the toolbar
    fig4.canvas.toolbar.pack_forget()

    #Display the detected face
    axs4.imshow(cv2.cvtColor(faces, cv2.COLOR_BGR2RGB),aspect='auto')
    plt.waitforbuttonpress(0)
    plt.close()
    
    #Call the function to recognise the emotion expressed by the face. and store it in a string
    emotion=recognise_emotion(location,faces)


    emostr='Emotion Detected :' + emotion
    
    fig5,axs5 = plt.subplots()
    axs5.axis('off')
    axs5.set_title(emostr)
    plt.tight_layout()
    figManager = plt.get_current_fig_manager()  
    figManager.full_screen_toggle()
    fig5.canvas.toolbar.pack_forget()
    #Displaying detected emotion
    axs5.imshow(cv2.cvtColor(faces, cv2.COLOR_BGR2RGB),aspect='auto')
    plt.waitforbuttonpress(0)
    plt.close()

    

    input('Press ENTER to exit')
    

