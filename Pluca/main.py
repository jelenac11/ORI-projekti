from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os

k = 0
for count, filename in enumerate(os.listdir("./chest_xray_data_set/TRAIN/BACTERIA")):
    img = load_img("./chest_xray_data_set/TRAIN/BACTERIA/" + filename)
    datagen = ImageDataGenerator(
        rotation_range=40,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)

    i = 1
    for batch in datagen.flow(x,
                              save_to_dir='C:/Users/PC/Desktop/bacteria2/', save_prefix='aaabacteria',
                              save_format='jpeg'):
        if i >= 2:
            break
        i += 1
        k += 1
    if k > 500:
        break
