# NotMnist
![NotMnist](http://yaroslavvb.com/upload/notMNIST/nmn.png)

基于TensorFlow r1.2 Python 3.6 实现的[字母A-J识别](http://yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html)

## 数据处理
- data.py 该程序将从网络中下载训练图。并导出一个pickle文件，这个文件最终可以随时读入内存并取得训练、验证、测试集数据来使用。
  1. 下载
  2. 解压缩
  3. 校验图片格式并打包成小的pickle
  4. 读取到内存
  5. 去除重复
  6. 随机打乱
  7. 打包为最终约690mb左右的pickle文件

 

## notMNIST.pickle
所有的数据集以保存在notMNIST.pickle方便读取
- Training set (200000, 28, 28) (200000,)
- Validation set (10000, 28, 28) (10000,)
- Test set (10000, 28, 28) (10000,)

## 训练模型，并导出模型
- train.py 训练模型并导出模型

## 使用图片测试识别效果
- train_test.py 导入以保存的模型，并以实图测试。