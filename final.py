#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
from imutils import contours as imutils_contours
from skimage import img_as_ubyte
from PIL import Image
from skimage.exposure import histogram
import imutils
import easyocr
import tkinter as tk
from PIL import Image, ImageTk
from tkinter.filedialog import askopenfilename


# In[ ]:


def show_images(images, titles=None):
    n_ims = len(images)
    if titles is None:
        titles = ['(%d)' % i for i in range(1, n_ims + 1)]
    fig = plt.figure()
    n = 1
    for image, title in zip(images, titles):
        a = fig.add_subplot(1, n_ims, n)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
        plt.axis('off')
        n += 1
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_ims)
    plt.show()


# In[ ]:


def isPossiblePlate(rect, img):
    (_, _), (width, height), _ = rect
    w = width
    h = height
    if(height>width):
        w = height
        h = width
    if h == 0 or w == 0 or w/h < 2:
        return False
    img_area = img.shape[0] * img.shape[1]
    area = w*h
    if area < 0.002*img_area or area > 0.75*img_area:
        return False
    return True


# In[ ]:


def preprocess(img, close_struct_size=26):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlurred = cv2.GaussianBlur(img, (7, 7), 0)
    sobelx = cv2.Sobel(imgBlurred, cv2.CV_8U, 1, 0, ksize=3)
    ret2, threshold_img = cv2.threshold(
        sobelx, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    element = cv2.getStructuringElement(
        shape=cv2.MORPH_RECT, ksize=(close_struct_size, 3))
    element2 = cv2.getStructuringElement(
        shape=cv2.MORPH_RECT, ksize=(10, 3))
    morph_n_thresholded_img = threshold_img.copy()
    img_intermediate = morph_n_thresholded_img
    cv2.morphologyEx(src=threshold_img, op=cv2.MORPH_CLOSE,
                     kernel=element, dst=img_intermediate)
    cv2.morphologyEx(src=img_intermediate, op=cv2.MORPH_OPEN,
                     kernel=element2, dst=morph_n_thresholded_img)
    kernel = np.ones((10, 10), 'uint8')
    morph_n_thresholded_img = cv2.erode(morph_n_thresholded_img, kernel)
    return morph_n_thresholded_img


# In[ ]:


def fixRange(img):
    max = np.amax(img)
    if(max <= 1):
        return img_as_ubyte(img)
    else:
        return img_as_ubyte(img/255)


# In[ ]:


def getLicensePlate(processed_img, img, big=False):
    contours, _ = cv2.findContours(processed_img, mode=cv2.RETR_EXTERNAL,
                                   method=cv2.CHAIN_APPROX_NONE)
    # io.imshow(cv2.drawContours(img.copy(), contours, -1, (0,255,0), 3))
    imgs = []
    for contour in contours:
        min_rect = cv2.minAreaRect(contour)
        if isPossiblePlate(min_rect, img):
            x, y, w, h = cv2.boundingRect(contour)
            after_validation_img = []
            if big:
                after_validation_img = img[y-20:y+20 + h, x-10:x+10 + w]
            else:
                after_validation_img = img[y-15:y+5 + h, x-5:x+5 + w]
            imgs.append(after_validation_img)
    return imgs


# In[ ]:


def extractCharactersFromPlate(image, resize_state=False, invert=True, threshold=150):
    image = imutils.resize(image, width=500)
    if resize_state:
        size = image.shape
        image = image[4:size[0]-5][4:size[1]-5]
    plate_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if invert:
        plate_gray = 255-plate_gray
    thresh = cv2.threshold(plate_gray, threshold, 255, cv2.THRESH_BINARY)[1]
    # kernel = np.ones((2, 3), 'uint8')
    # thresh = cv2.erode(thresh, kernel)
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # show_images([thresh, cv2.drawContours(
    #     image.copy(), contours, -1, (0, 255, 0), 3)])
    contours = imutils_contours.sort_contours(
        contours, method="left-to-right")[0]
    output = []
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        part = thresh[y-1:y+1 + h, x-1:x+1 + w]
        if h/w > 1 and h/w < 7:
            try:
                part = thresh[y-1:y+1 + h, x-1:x+1 + w]
                if len(histogram(part)[0]) < 2:
                    continue
                Level = 100
                part = np.hstack(
                    (np.zeros((part.shape[0], 20)), part,  np.zeros((part.shape[0], 20))))
                part = np.vstack(
                    (np.zeros((20, part.shape[1])), part,  np.zeros((20, part.shape[1]))))
                output.append(part.astype('uint8'))
            except:
                continue
    return thresh, output


# In[ ]:


reader = easyocr.Reader(['en'])
def getStringFromCharArrays(chars):
    new_chars = []
    string = ""
    for i in chars:
        # i = resize(i, (400, 200))
        # i = np.where(i == 0, 0, 1)
        # i = binary_opening(i)
        # i = img_as_ubyte(i)
        # kernel = np.ones((15,15), 'uint8')
        # i = cv2.erode(i, kernel)
        # i = Image.fromarray(np.uint8(i*255))
        # i = i.filter(ImageFilter.ModeFilter(size=20))
        # i = np.array(i)
        # kernel = np.ones((10,10), 'uint8')
        # i = cv2.dilate(img_as_ubyte(i), kernel)
        i_uint8 = cv2.normalize(
            i, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        new_chars.append(i_uint8)
        alphanumeric = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        char = reader.readtext(i_uint8, detail=0, allowlist=alphanumeric)
        # char = pytesseract.image_to_string(
            # i_uint8, lang='eng', config='--psm 10')
        if len(char) > 0:
            string += char[0]
    return string, new_chars


# In[ ]:


def getStringFromPlate(plate, threshold=120):
    chars = []
    chars = extractCharactersFromPlate(plate, threshold)[1]
    if len(chars) < 4:
        chars = extractCharactersFromPlate(
            plate, threshold=170, resize_state=True)[1]
    if len(chars) < 4:
        chars = extractCharactersFromPlate(
            plate, threshold=170, invert=False, resize_state=True)[1]
    # show_images(chars)
    string, new_chars = getStringFromCharArrays(chars)
    return string, new_chars, chars


# In[ ]:


def readPlate(img):
    processed_img = preprocess(img)
    plates = getLicensePlate(processed_img, img)
    # show_images([img, processed_img])
    for p in plates:
        try:
            string1, new_chars, chars = getStringFromPlate(p)
            # show_images(chars)
            # show_images(new_chars)
            if string1:
               return string1
        except:
            continue
    processed_img = preprocess(img, close_struct_size=35)
    plates = getLicensePlate(processed_img, img, big=True)
    # show_images([img, processed_img])
    for p in plates:
        try:
            string2, new_chars, chars = getStringFromPlate(p)
            # show_images(chars)
            # show_images(new_chars)
            if string2:
               return string2
        except:
            continue
    return "Couldn't get any characters"


# In[ ]:


def open_file():
    browse_text.set("loading...")
    imgpath = askopenfilename(parent=root, title="Choose an Image", filetypes=[("Images", ".png .jpeg")])
    if imgpath:
        browse_text.set("Processing...")
        img = io.imread(imgpath)
        string = readPlate(img)
        if string != "Couldn't get any characters":
            string="a licence plate with numbers " + string +  " was found"
        text_box = tk.Text(root, height=10, width=50, padx=15, pady=15)
        text_box.insert(1.0, string)
        text_box.tag_configure("center", justify="center")
        text_box.tag_add("center", 1.0, "end")
        text_box.grid(column=1, row=3)
        browse_text.set("Browse")


# In[ ]:


root = tk.Tk()
canvas = tk.Canvas(root, width=600, height=300)
canvas.grid(columnspan=3, rowspan=3)
#logo
logo = Image.open('logo.png')
logo = ImageTk.PhotoImage(logo)
logo_label = tk.Label(image=logo)
logo_label.image = logo
logo_label.grid(column=1, row=0)

#instructions
instructions = tk.Label(root, text="Select an image from your computer", font="Raleway")
instructions.grid(columnspan=3, column=0, row=1)


#browse button
browse_text = tk.StringVar()
browse_btn = tk.Button(root, textvariable=browse_text, command=lambda:open_file(), font="Raleway", bg="#20bebe", fg="white", height=2, width=15)
browse_text.set("Browse")
browse_btn.grid(column=1, row=2)


canvas = tk.Canvas(root, width=600, height=250)
canvas.grid(columnspan=3)

root.mainloop()

