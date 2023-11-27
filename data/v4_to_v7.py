import os

path_ = ["train","valid","test"]
dataset = 0
num = 0

os.chdir(os.path.join("delivery_sign_0725_part"))
for dir in path_:
    os.chdir(os.path.join(dir))
    try:
        os.mkdir('images')
        os.mkdir('labels')
    except:
        pass
    for txtfile in os.listdir(os.getcwd()):
        if txtfile.endswith(".txt"):
            dataset +=1
    os.chdir("..")

split_set =[int(0.7 * dataset),int(0.9 * dataset)]
print(split_set)

for dir in path_:
    os.chdir(os.path.join(dir))
    for txtfile in os.listdir(os.getcwd()):
        if txtfile.endswith(".txt"):
            basename=os.path.basename(txtfile)
            filename=os.path.splitext(basename)[0]
            image_path = filename + '.jpg'
            text_file_path = filename + '.txt'
            
            os.chdir("..")
            if num < split_set[0]:
                os.rename(dir + '/' + image_path, 'train/images/train_' + str(num) + '.jpg')
                os.rename(dir + '/' + text_file_path, 'train/labels/train_' + str(num) + '.txt')

            elif num < split_set[1]:
                os.rename(dir + '/' + image_path, 'valid/images/valid_' + str(num - split_set[0]) + '.jpg')
                os.rename(dir + '/' + text_file_path, 'valid/labels/valid_' + str(num - split_set[0]) + '.txt')

            else:
                os.rename(dir + '/' + image_path, 'test/images/test_' + str(num - split_set[1]) + '.jpg')
                os.rename(dir + '/' + text_file_path, 'test/labels/test_' + str(num - split_set[1]) + '.txt')
            
            os.chdir(os.path.join(dir))
            num += 1
    os.chdir("..")