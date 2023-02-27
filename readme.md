## 动物识别系统使用说明

### 文件目录

- 动物识别项目
  - configs
    - config.yaml配置文件
  - demo：用训练好的模型去测试自己的图片
    - input：随意放入动物图片
    - output：输出result.csv，分类结果
  - experiments：运行过程中用的代码
    - check_error.py：数据集可能错误，检查&剔除打不开的图片文件
    - dataloader.py：加载数据集，返回dataloader
    - test.py：测试模型（对demo文件夹进行操作）
    - train.py：训练模型
  - models：神经网络模型
    - resnet.py：一个基于resnet的fine-tuning的网络，输出10个种类判别
  - demo.py：对demo文件夹操作，加载训练好的model_static_dict.pth模型，并进行图片分类测试
  - model_static_dict.pth：保存下来的训练好的模型。已经在服务器上对模型训练了50个epoch，精度达到98.9%
  - run.py：用于训练网络
  - logs：日志文件，可以看到之前的训练过程记录
  - data（数据集存放文件夹，自己建立）
    - Data-V2（数据集，解压后得到，下载地址：[Animals - V2 | Image Classification Dataset | | Kaggle](https://www.kaggle.com/datasets/utkarshsaxenadn/animal-image-classification-dataset?resource=download-directory&select=Animal-Data-V2)
      - Interesting Data
      - Testing Data
      - TFRecords
      - Training Data
      - Validation Data

### 使用方法：

1. 根据需要调整config.yaml
2. 如果想自己训练模型，运行run.py；如果想直接用我训练好的，跳过这一步
3. 在demo文件夹的input中放入测试图片（已经从原数据集提供的testdata中每类动物挑选4张图片放入）
4. 运行demo.py
5. 查看demo文件夹中的output中的输出文件，可以看到测试图片文件对应的分类结果

*注意：数据集选择Data-V2版本*![数据集下载界面](E:\文档\大四上-课程\学AI\动物识别项目\数据集下载界面.jpg)