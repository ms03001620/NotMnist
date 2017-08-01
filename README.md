# NotMnist
![NotMnist](http://yaroslavvb.com/upload/notMNIST/nmn.png)

基于TensorFlow r1.2 Python 3.6 实现的[字母A-J识别](http://yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html)

## 性能
- 测试集准确度达到96.2% 训练集batch为200 训练时间长度2小时 学习率0.02 总学习次数为20万次

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


## License

    Copyright (C) 2017 zhenjin ma

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
