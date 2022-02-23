import numpy as np

def finite_generator(data, batch_size):
    x, y = data
    i = 0 # Type: int
    while True:
        j = i + batch_size
        # If we wrap around the back of the dataset:
        if j >= x.shape[0]:
            yield (x[i:x.shape[0],...], y[i:x.shape[0]])
            break
        else:
            yield (x[i:j,...], y[i:j])
            i = j