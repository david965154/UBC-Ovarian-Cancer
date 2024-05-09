# UBC Ovarian Cancer Subtype Classification and Outlier Detection (UBC-OCEAN)

[Competition website](https://www.kaggle.com/competitions/UBC-OCEAN)

競賽資料集包含了五類卵巢癌亞型的 Whole Slide Image (WSI)，其中也含了 thumbnail 版本，為 WSI 的較低解析度版本，另外 tma 則為不同放大倍率版本，此種圖每一類各給 5 張。最後期望是能分類出五類並能檢測出異常的亞種，即沒有看過這種類型的腫瘤就將他歸為 others ，而其他亞種是模型完全沒看過的。

WSI segmentation :
原先切割方式為透過圖像顏色數值占比做切割並篩去背景及組織間空洞部分，但事實上有些組織顏色會因為太淡的太多而容易被歸類為 threshold 想要篩去的部分，有些圖像則是因為偏暗則無法被篩去，這其實會大幅影響資料集品質。因此改以顏色分布的方式篩選，若同一顏色占比太高則刪去，這樣不會因為組織顏色太淡或太深就被篩去，而可以單純刪去背景佔太多的部分而避開無法適用於所有圖的 threshold 。再來因為 tma 圖像太少，因此切出圖的中間部分並分割成四份可以產出更多的 tma 圖且在 resize 時不會犧牲圖像解析度。這裡直接輸出腫瘤占比以得出這邊切出來的 patch 有多少是含腫瘤的。


Training :
- augmentation
  - train : Resize, ShiftScaleRotate, HueSaturationValue, RandomBrightnessContrast, Normalize
  - validation, test : Resize, Normalize
- batch_size : 16
- use_amp : True 混合精度 : 將數據以 float16 及 float32 儲存，可大幅減少 GPU ram 的使用
- epochs : 30 
- lr : 0.0001
- num_classes : 6
- CLIP_GRAD : 5.0 梯度截斷，減緩梯度爆炸，若梯度超過一定值則會透過操作改變梯度值
- seed : 42
- image_size : 1024
- Pooling layer : GeM 透過同時使用 Average Pooling 及 Max Pooling 達到同時關注局部及全局
- entropy loss : CrossEntropy
- optimizer : AdamW
- CosineAnnealingLR 調整 lr

