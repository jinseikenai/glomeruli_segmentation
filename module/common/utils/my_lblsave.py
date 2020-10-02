import os
import numpy as np
import PIL.Image

from labelme import logger
from labelme.utils.draw import label_colormap

from matplotlib import pyplot as plt
from matplotlib import gridspec


def lblsave(filename, lbl, size=None):
    if os.path.splitext(filename)[1] not in ['.png', '.PNG']:
        filename += '.png'
    # Assume label ranses [-1, 254] for int32,
    # and [0, 255] for uint8 as VOC.
    if lbl.min() >= -1 and lbl.max() < 255:
        lbl_pil = PIL.Image.fromarray(lbl.astype(np.uint8), mode='P')
        if size is not None:
            if isinstance(size, tuple):
                lbl_pil = lbl_pil.resize(size)
            else:
                raise AttributeError('size is not set properly. given size:{}'.format(size))

        colormap = (label_colormap(255) * 255).astype(np.uint8)
        lbl_pil.putpalette(colormap.flatten())
        lbl_pil.save(filename)
    else:
        logger.warn(
            '[%s] Cannot save the pixel-wise class label as PNG, '
            'so please use the npy file.' % filename
        )


def org_lbl_save(filename, org, label):
    if os.path.splitext(filename)[1] not in ['.png', '.PNG']:
        filename += '.png'
    # Assume label ranses [-1, 254] for int32,
    # and [0, 255] for uint8 as VOC.
    if label.min() >= -1 and label.max() < 255:

        org_img = PIL.Image.fromarray(org.astype(np.uint8), mode='RGB')
        lbl_pil = PIL.Image.fromarray(label.astype(np.uint8), mode='P')
        colormap = (label_colormap(255) * 255).astype(np.uint8)
        lbl_pil.putpalette(colormap.flatten())

        plt.figure(figsize=(13, 10))
        grid_spec = gridspec.GridSpec(2, 2, width_ratios=[5, 5], height_ratios=[8, 2])

        plt.subplot(grid_spec[0])
        plt.imshow(org_img)
        plt.axis('off')
        plt.title('input image')

        plt.subplot(grid_spec[1])
        plt.imshow(lbl_pil)
        plt.axis('off')
        plt.title('GT(label image)')

        label_names, full_color_map = get_label_name_and_map(colormap)
        unique_labels = range(0, len(label_names))
        ax = plt.subplot(grid_spec[3])
        plt.imshow(
            full_color_map[unique_labels].astype(np.uint8), interpolation='nearest')
        ax.yaxis.tick_right()
        plt.yticks(range(len(unique_labels)), label_names[unique_labels])
        plt.xticks([], [])
        ax.tick_params(width=0.0)
        plt.grid('off')

        plt.savefig(filename)
        plt.close()
    else:
        logger.warn(
            '[%s] Cannot save the pixel-wise class label as PNG, '
            'so please use the npy file.' % filename
        )


def get_label_name_and_map(colormap):
    label_names = np.array([
        'background', 'glomerulus', 'crescent', 'collapsing/sclerosis', 'mesangium'
    ])

    full_label_map = np.arange(len(label_names)).reshape(len(label_names), 1)
    return label_names, label_to_color_image(full_label_map, colormap)


def label_to_color_image(label, colormap):
    """Adds color defined by the dataset colormap to the label.

    Args:
      label: A 2D array with integer type, storing the segmentation label.

    Returns:
      result: A 2D array with floating type. The element of the array
        is the color indexed by the corresponding element in the input label
        to the PASCAL color map.

    Raises:
      ValueError: If label is not of rank 2 or its value is larger than color
        map maximum entry.
    """
    if label.ndim != 2:
        raise ValueError('Expect 2-D input label')

    if np.max(label) >= len(colormap):
        raise ValueError('label value too large.')

    return colormap[label]

