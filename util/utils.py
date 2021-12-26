def compare_self_for_muti(path):
    with open(path, encoding='gbk') as f:
        begin_txt=[]
        for line in f:
            begin_txt.append(line.strip())
    next_txt = begin_txt
    count_same = 0
    for i in next_txt:
        for j in begin_txt:
            if i == j:
                count_same += 1
    count_same = count_same-len(next_txt)
    return count_same

def change_image_shape(images):
    shape_tuple = images.shape
    if len(shape_tuple) == 3:
        images = images.reshape(-1, shape_tuple[-1], shape_tuple[-1], 1)
    elif shape_tuple == 4 and shape_tuple[-1] > 3:
        images = images.reshape(-1, shape_tuple[-1], shape_tuple[-1], shape_tuple[1])
    return images
