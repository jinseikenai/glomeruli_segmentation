import numpy as np
import PIL.Image
import PIL.ImageDraw

from labelme import logger


def polygons_to_mask_base(img_shape, polygons):
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    mask = PIL.Image.fromarray(mask)
    xy = list(map(tuple, polygons))
    PIL.ImageDraw.Draw(mask).polygon(xy=xy, outline=1, fill=1)
    return mask


def polygons_to_mask(img_shape, polygons):
    mask = np.array(polygons_to_mask_base(img_shape, polygons), dtype=bool)
    return mask

def shapes_to_label(img_shape, shapes, label_name_to_value, label_name_list, type='class'):
    assert type in ['class', 'instance']

    cls = np.zeros(img_shape[:2], dtype=np.int32)
    if type == 'instance':
        ins = np.zeros(img_shape[:2], dtype=np.int32)
        instance_names = ['_background_']
    for label_name in label_name_list:
        target_shapes = [x for x in shapes if x['label'] == label_name]
        for shape in target_shapes:
            polygons = shape['points']
            label = shape['label']
            if type == 'class':
                cls_name = label
            elif type == 'instance':
                cls_name = label.split('-')[0]
                if label not in instance_names:
                    instance_names.append(label)
                ins_id = len(instance_names) - 1
            cls_id = label_name_to_value[cls_name]
            mask = polygons_to_mask(img_shape[:2], polygons)
            cls[mask] = cls_id
            if type == 'instance':
                ins[mask] = ins_id

    if type == 'instance':
        return cls, ins
    return cls
