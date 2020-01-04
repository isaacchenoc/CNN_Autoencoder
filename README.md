# CNN_Autoencoder

Two different types of CNN auto encoder, implemented using pytorch. One has only convolutional layers and other consists of convolutional layers, pooling layers, flatter and full connection layers. These two auto encoders were implemented as I wanted to see how pooling layers, flatter and full connection layers can affect the efficiency and the loss of data compression. As such, a normal CNN was also implemented for classification. The dataset MNIST was used because of its simplicity. Similar datasets such as cifar-10 should work in this case as well. If you find interested in this topic, please read the detailed report, as I have captured the process and result of the comparison. 

## How to run
to run any of the python files, make sure the 'data' folder is in the same directory

cnn.py includes the CNN that classify MNIST. learning_rate, batch_size and num_epochs can be changed to different value at the beginning of the file.

cnn_ae1.py includes Auto encoder 1 to encode and decode MNIST and a CNN that takes the restructured data as input to make  classification. learning_rate, batch_size_ae, batch_size, num_epochs_ae, num_epochs can be changed at the beginning of the file, where batch_size_ae and num_epochs_ae are for AE 1 and batch_size and num_epochs are for the CNN. the methods for showing images are currently in comment, so if want the images to be showed, remove the """ in line 94 and 121. 

cnn_ae2.py includes Auto encoder 2 to encode and decode MNIST and a CNN that takes the restructured data as input to make  classification. learning_rate, batch_size_ae, batch_size, num_epochs_ae, num_epochs can be changed at the beginning of the file, where batch_size_ae and num_epochs_ae are for AE 2 and batch_size and num_epochs are for the CNN. the methods for showing images are currently in comment, so if want the images to be showed, remove the """ in line 91 and 109.