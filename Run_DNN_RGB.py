Python 3.9.1 (v3.9.1:1e5d33e9b9, Dec  7 2020, 12:10:52) 
[Clang 6.0 (clang-600.0.57)] on darwin
Type "help", "copyright", "credits" or "license()" for more information.
>>> 
========= RESTART: /Users/mariam/Desktop/untitled folder 2/RGB&GRAY.py =========
#############################################################################
The color:BRG
1408
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 98, 98, 100)       2800      
                                                                 
 conv2d_1 (Conv2D)           (None, 96, 96, 50)        45050     
                                                                 
 max_pooling2d (MaxPooling2D  (None, 32, 32, 50)       0         
 )                                                               
                                                                 
 conv2d_2 (Conv2D)           (None, 30, 30, 100)       45100     
                                                                 
 conv2d_3 (Conv2D)           (None, 28, 28, 120)       108120    
                                                                 
 max_pooling2d_1 (MaxPooling  (None, 14, 14, 120)      0         
 2D)                                                             
                                                                 
 conv2d_4 (Conv2D)           (None, 12, 12, 75)        81075     
                                                                 
 conv2d_5 (Conv2D)           (None, 10, 10, 50)        33800     
                                                                 
 max_pooling2d_2 (MaxPooling  (None, 3, 3, 50)         0         
 2D)                                                             
                                                                 
 flatten (Flatten)           (None, 450)               0         
                                                                 
 dense (Dense)               (None, 50)                22550     
                                                                 
 dense_1 (Dense)             (None, 29)                1479      
                                                                 
=================================================================
Total params: 339,974
Trainable params: 339,974
Non-trainable params: 0
_________________________________________________________________
Epoch 1/10

Epoch 2/10

Epoch 3/10

Epoch 4/10

Epoch 5/10

Epoch 6/10

Epoch 7/10

Epoch 8/10

Epoch 9/10

Epoch 10/10


Accuracy: 0.5673758865248227
Precision: [1.         0.77777778 1.         0.71428571 0.38888889 0.35294118
 0.73333333 0.6        0.75       0.34782609 0.37037037 0.625
 0.69230769 0.5        0.60869565 1.         0.8        0.625
 0.5        0.6        0.54545455 0.5        0.30555556 0.25
 0.20833333 0.45454545 1.         1.         0.7       ]
Recall: [0.75       0.875      0.53333333 0.625      0.41176471 0.75
 0.64705882 1.         0.6        0.88888889 0.83333333 0.5
 0.64285714 0.26666667 0.7        0.46666667 0.66666667 0.33333333
 0.15789474 0.8        0.33333333 0.625      0.78571429 0.04166667
 0.38461538 0.29411765 1.         0.76923077 0.63636364]
>>> 