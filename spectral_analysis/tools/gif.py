### Taken from https://github.com/maxhumber/gif/blob/master/gif.py
import functools
import io

from matplotlib import pyplot as plt
from PIL import Image

def frame(func):
    """
    Decorator for a matplotlib plot function.
    Example:
    ```
    @gif.frame
    def plot(x, y):
        plt.figure(figsize=(5, 5))
        plt.scatter(x, y)
        plt.xlim((0, 100))
        plt.ylim((0, 100))
    ```
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        buffer = io.BytesIO()
        func(*args, **kwargs)
        plt.savefig(buffer, format="png")
        buffer.seek(0)
        image = Image.open(buffer)
        plt.close()
        return image

    return wrapper


def save(frames, path, duration=100):
    """
    Save decorated frames to an animated gif.
    - frames (list): collection of frames built with the frame decorator
    - path (str): filename with relative or absolute path
    - duration (int): milliseconds between frames
    """
    frames = iter(frames)
    firstframe = next(frames)

    firstframe.save(
        path,
        save_all=True,
        append_images=frames,	# Esto excluye el primero, ya que fue leído usando next()
        optimize=True,
        duration=duration,
        loop=0,
    )

