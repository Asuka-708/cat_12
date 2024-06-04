###模型的介绍    
  
此项目基于ResNet50迁移学习实现对12种猫的分类        
  
其中数据集来自视觉中国和Kaggle以及百度飞浆通过收集和修改制成    
  
数据集的0，1，2，3，4，5，6，7，8，9，10，11，12分别为    
    0: '阿比西亚猫',                   
    1: '孟加拉豹猫',    
    2: '伯曼猫',     
    3: '孟买猫',    
    4: '英国短毛猫',     
    5: '埃及猫',    
    6: '缅因猫',    
    7: '波斯猫',     
    8: '布偶猫',                            
    9: '俄罗斯蓝猫',      
    10: '暹罗猫',               
    11:'无毛猫'    
项目包括数据集的分类，通过模型训练，模型预测，生成比赛所用的csv文件    
  
模型优化基于SGD（随机梯度下降）来进行以防止过拟合     
  
模型通过数据增强来提高泛化性如随机翻转，随机色调，亮度，饱和度，对比度等来实现数据增强，以提高泛化性    
  
超参数在开头以便对模型进行调参，提高模型准确度     
  
通过对结果的对吧的方式来记录不同参数对结果的影响以找到最适参数。  
  
此模型还用到tqdm以实现训练模型进度的可视化，显示训练时间，训练速度，分别返回训练真实值和测试真实值。    

