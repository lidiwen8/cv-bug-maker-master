# 使用VGG19实现口罩及社交距离检测

社交距离，也称为“物理距离”，是指在您和其他非您家庭的人之间保持安全空间。

我们的项目目标是建立一个深度学习模型，该模型可以识别该人是否戴口罩，还可以检测人们是否违反了社交距离规范。并且该程序不仅能识别静态图像，也能实时调用摄像头来进行检测。

选题报告：https://gitee.com/yinzhi-code/cv-bug-maker/blob/master/%E9%80%89%E9%A2%98%E6%8A%A5%E5%91%8A.md

数据集链接：
* https://www.kaggle.com/andrewmvd/face-mask-detection
* https://www.kaggle.com/ashishjangra27/face-mask-12k-images-dataset
* https://www.kaggle.com/lalitharajesh/haarcascades

项目报告：https://gitee.com/yinzhi-code/cv-bug-maker/blob/master/%E9%A1%B9%E7%9B%AE%E6%8A%A5%E5%91%8A.md

B站视频：https://www.bilibili.com/video/BV1zS4y1T7PN/

model.py:训练生成模型，存储在masknet.h5中

main_staticImage.py:识别静态图像

main_video.py:通过摄像头动态识别

