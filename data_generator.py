from matplotlib import image
import numpy as np
import sklearn

def data_generate(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        samples = sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                center_name = batch_sample[0]
                center_image = image.imread(center_name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

                center_image_flipped = np.fliplr(center_image)
                center_angle_flipped = -center_angle
                images.append(center_image_flipped)
                angles.append(center_angle_flipped)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)
