'''
此代码是查找出一个文件夹里面，
所有图片读取错误，再删除
'''
import os
import time

import warnings

from PIL import Image
class checkErrorImgs():
    def __init__(self,root_dir):
        self.base_dir=root_dir
        warnings.filterwarnings("error", category=UserWarning)

    def is_read_successfully(self,file):
        try:
            imgFile = Image.open(file)  # 这个就是一个简单的打开成功与否
            return True
        except Exception:
            return False

    def check(self):
        errorImgs=[]
        for parent, dirs, files in os.walk(self.base_dir):#(root,dirs,files)
            print("\nparent\t",parent,"\ndirs\t",dirs,"\nfiles\t",files,"\n")
            time.sleep(1)
            for file in files:
                if not self.is_read_successfully(os.path.join(parent, file)):
                    errorImgs.append(os.path.join(parent, file))
                    print(os.path.join(parent, file))
        return errorImgs
    def doDelete(self):
        errorImgs=self.check()
        for i in errorImgs:
            os.remove(i) #真正使用时，这一行要放开，自己一般习惯先跑一遍，没有错误了再删除，防止删错。

checkErrorImgs("../../data/Data-V2").check()