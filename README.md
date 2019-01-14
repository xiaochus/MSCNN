# MSCNN
A Python 3 and Keras 2 implementation of MSCNN for people countingand provide train method.

## Requirement
- Python 3.6 
- Keras 2.2.2
- Tensorflow-gpu 1.8.0  
- OpenCV 3.4

## MSCNN and MSB architectures
**MSCNN**  

![MSCNN](/images/mscnn.png)

**MSB**  

![MSB](/images/msb.png)

## Experiment

**data**

[Mall Dataset crowd counting dataset](http://personal.ie.cuhk.edu.hk/~ccloy/downloads_mall_dataset.html)

Generate density_map from data：
![density map](/images/density_map.png)

**train**

run the follow command:
```
python train.py --size 224 --batch 16 --epochs 100
```

## Reference

	@article{MSCNN,  
	  title={Multi-scale convolutional neural network for crowd counting},  
	  author={Lingke Zeng, Xiangmin Xu, Bolun Cai, Suo Qiu, Tong Zhang},
	  journal={2017 IEEE International Conference on Image Processing (ICIP)},
	  year={2017}
	}


## Copyright
See [LICENSE](LICENSE) for details.
 
