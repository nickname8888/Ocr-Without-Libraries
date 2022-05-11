# Ocr-Without-Libraries

The goal of this repository is to achieve character recognition withouot using any machine learning libraries and building everything from scratch. I have implemented a basic k-nearest neighbours classifier to recognize numeric digits 0-9. 


<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/2/27/MnistExamples.png/320px-MnistExamples.png" width="500" height="300">
<img src="https://www.theclickreader.com/wp-content/uploads/2020/08/23-1024x576.png" width="500" height="300">

* The dataset used here is the MNIST dataset, comprising of 60,000 images of different nunmeric characters for the training set and 10,000 images for the  testing set. 
* The k-nearest neighbours algorithm is used to classify between different characters. 
* The 28x28 images are first flattened to form a one dimensional array consisting of 784 pixel values. 
* They are then plotted and clusters are formed based on their positioning on the graphs. 
* WHen a new entry is plotted to the graph, the euclidian distance between its "k" nearest points is calculates. Then a simple count operation is performed to determine which cluster the new point should belong to. 
* This is the simple working of KNN for our use case. This implementation can be further extended to different datasets since the core concepts are same across all problems.  
