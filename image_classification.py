import os, shutil


def copy_images(image_list, origin_path, destination_path):
    """
    Takes in:
        image_list = list of image file names
        origin_path = path where the images are saved
        destination_path = path where the images will be copied to
    """
    #checks if the origin path exists
    if os.path.exists(origin_path) == False:
        print("incorrect origin path")

    #checks if the destination path exists, if not then create
    if os.path.exists("dataset/") == False:
        os.mkdir("dataset/")
    if os.path.exists("dataset/images/") == False:
        os.mkdir("dataset/images/")
    if os.path.exists("dataset/images_unique/") == False:
        os.mkdir("dataset/images_unique/")
    if os.path.exists(destination_path) == False:
        os.mkdir(destination_path)

    #copy each image from origin to destination folder
    for img in image_list:
        origin = os.path.join(origin_path, img)
        destination = os.path.join(destination_path, img)
        shutil.copyfile(origin, destination)

    print("done copying images to " + destination_path)