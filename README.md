# NotMnist

基于TensorFlow r1.2 实现的[字母A-J识别](http://yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html)

## notMNIST.pickle
所有的数据集以保存在notMNIST.pickle方便读取
- Training set (200000, 28, 28) (200000,)
- Validation set (10000, 28, 28) (10000,)
- Test set (10000, 28, 28) (10000,)

## 训练模型，并导出模型
- train.py 训练模型并导出模型

## 使用图片测试识别效果
- train_test.py 导入以保存的模型，并以实图测试。