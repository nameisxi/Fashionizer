import numpy as np
import math
from PIL import Image
import cv2
from matplotlib import pyplot as plt


class MnistCreator():
    def get_as_png(self, file_name, save_as):
        '''Converts the input image to PNG format.'''
        image = Image.open(file_name)
        image.save(save_as)
        image = Image.open(save_as)
        return image

    def crop(self, image_data):
        '''"Crops" image by changing all RGB values to [255, 255, 255] if they differ more than
            70% from the pixel in the center of the image.
        '''
        image_data.setflags(write=1)
        height, width, channels = image_data.shape
        new_image_data = np.full((height, width, 3), 255)
        middle_pixel = image_data[(height // 2), (width // 2)]
        middle_pixel_avg = np.mean(middle_pixel)
        difference_limit = middle_pixel_avg * 0.7
        for row in range(height):
            for col in range(width):
                pixel_avg = np.mean(image_data[row, col])
                if (abs(middle_pixel_avg - pixel_avg) <= difference_limit):
                    new_image_data[row, col] = image_data[row, col]
        return new_image_data

    def trim(self, image_data):
        '''Trimmes rows and columns that are fully white e.g. contain only arrays [255, 255, 255].'''
        height, width, channels = image_data.shape   
        trimmed = np.copy(image_data)
        indexes = []
        for i in range(height):
            mean = np.mean(image_data[i, :])
            if mean == 255.0:
                indexes.append(i)
        trimmed = np.delete(trimmed, indexes, axis=0)
    
        height, width, channels = trimmed.shape
        indexes = []
        for i in range(width):
            mean = np.mean(trimmed[:, i])
            if mean == 255.0:
                indexes.append(i)
        trimmed = np.delete(trimmed, indexes, axis=1)
        return trimmed

    def resize_longest_edge(self, file_name):
        '''Resizes the longest edge to 28 pixels by subsampling the pixels and
            scales the shorter edge accordingly.
        '''
        image = cv2.imread(file_name)
        height, width, channels = image.shape
        if height >= width:
            new_width = int((width / height) * 28)
            image = cv2.resize(image, (new_width, 28), interpolation=cv2.INTER_AREA)
        else:
            new_height = int((height / width) * 28)
            image = cv2.resize(image, (28, new_height), interpolation=cv2.INTER_AREA)
        return image 

    def extend_shortest_edge(self, image, image_data):
        '''Extends the shortest edge to 28 pixels and centers the new image.'''
        height, width, channels = image.shape
        upper_left_pixel = image_data[0, 0]
        upper_right_pixel = image_data[0, width - 1]
        background = np.zeros((28, 28, 3), np.uint8)
        background[:, 0:28 // 2] = (upper_left_pixel[0],upper_left_pixel[1],upper_left_pixel[2])
        background[:,28 // 2:] = (upper_right_pixel[0],upper_right_pixel[1],upper_right_pixel[2])
        x_offset = (28 - width) // 2
        y_offset = (28 - height) // 2
        background[y_offset:(y_offset + height), x_offset:(x_offset + width)] = image
        return background

    def negate_intensities(self, image_data):
        '''Negates pixel intensities in image.'''
        height, width, channels = image_data.shape
        for row in range(height):
            for col in range(width):
                red = 255 - image_data[row, col][0]
                green = 255 - image_data[row, col][1]
                blue = 255 - image_data[row, col][2]
                image_data[row, col] = [red,green,blue]
        return image_data 

    def convert_to_grayscale(self, image):
        '''Converts given pixels to 8-bit grayscale pixels.'''
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 

