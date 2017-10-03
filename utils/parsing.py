import os
import  glob


def parse_images_params_str(param_str):
    parse_list = param_str.split(':')
    images_dir = parse_list[0]
    if not os.path.isdir(images_dir):
        err_str = 'Failed to find dir: {} in parameter: {}'.format(images_dir, param_str)
        raise Exception(err_str)

    images_file = None
    if len(parse_list) > 1:
        # Images specified by paramstr being of the form A:B where A is a file with names of
        # image files while B is a directory where those images are
        images_file = parse_list[1]
        if not os.path.isfile(images_file):
            err_str = 'Failed to find file: {} in parameter: {}'.format(images_file,param_str)
            raise Exception(err_str)

    return images_dir, images_file

def get_full_names_of_image_files(images_dir, images_file=None):
    if images_file is not None:
        # A list of files, stored in in_images_file where each line contains a name of
        # a file in directory in_dir
        image_fnames = []  # LIst of full image file names for training
        with open(images_file, 'r') as fin:
            for line in fin.readlines():
                img_base_fname = line.strip()
                img_fname = os.path.join(images_dir, img_base_fname)
                if not os.path.isfile(img_fname):
                    err_str = 'Failed to find image file: {}'.format(img_fname)
                    raise Exception(err_str)
                image_fnames.append(img_fname)
        return image_fnames
    else:
        # Only directory is given --> take all files in this directory
        return get_names_of_image_files_in_dir(images_dir)

def get_names_of_image_files_in_dir(image_dir):
    """
    Get names of all image files in a directory.

    :param image_dir: Directory with images
    :return: A (sorted) list of names of all image files in imega_dir
    """
    IMG_TYPES = ('/*.jpg', '/*.png', '/*.JPG')
    image_fnames = []
    for type in IMG_TYPES:
        image_fnames.extend(glob.glob(image_dir + type))
    image_fnames.extend(glob.glob(image_dir))
    return image_fnames
