### Word2ec Python实现

#### Quick Start
```python
python main.py fpath './data/en.txt' save_path './output/cbow_zh_vectors.txt
```

#### 参数说明

|**字段**|**说明**|**数据类型**|
|---:|:---|:---|
|`fpath`|输入文件路径|str|    
|`save_path`|向量保存路径|str|  
|`cbow`|是否使用cbow，False表示使用SkipGram，默认True|bool|  
|`neg`|负采样个数，0表示不使用负采样，默认5|int|  
|`dim`|向量维度，默认100|int|  
|`alpha`|学习率，默认0.025|float|  
|`win`|窗口大小，默认5|int|  
|`min_count`|最小词频，默认5|int|  

#### 参考文献
> [1] 


    