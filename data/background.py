import os
import shutil

dataset = "delivery_unify"

os.chdir(os.path.join(dataset, "train", "labels"))

for txtfile in os.listdir(os.getcwd()):
    if txtfile.endswith(".txt"):
        f = open(txtfile)
        lines = f.readlines(0)

        if lines == []:
            basename=os.path.basename(txtfile)
            filename=os.path.splitext(basename)[0]
            os.chdir("..")
            os.remove('images/'+filename+'.jpg')
            os.remove('labels/'+filename+'.txt')
            os.chdir(os.path.join("labels"))

trainset_num = 0
for txtfile in os.listdir(os.getcwd()):
    if txtfile.endswith(".txt"):
        trainset_num += 1
print('Train Set Num = ' + str(trainset_num))

os.chdir("..")
os.chdir("..")
os.chdir("..")

for i in range(int(0.1*trainset_num)):
    shutil.copy('background.jpg',dataset+'/train/images/background_'+str(i)+'.jpg')
    shutil.copy('background.txt',dataset+'/train/labels/background_'+str(i)+'.txt')