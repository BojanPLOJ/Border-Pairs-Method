# Border-Pairs-Method
Border Pairs Method (BPM) is a machine learning (ML) algorithm intendent for Neural Networks (NN) learning. 
It is supervised classification algorithm for feedforward NN like Multilayer perceptron (MLP).
The most common algorithm for such NN is famouse Backpropagation (BP) which is far from ideal.
BP is iterative, nonconstructive, it find local minimum, it uses random values, user have to choose many hiperparameters...

BPM have many advantages over the BP:

     1) It finds near optimal NN construction
     2) It is done in one step without iterations
     3) It finds solution in each atempt
     4) It uses only useful patterns
     5) No need for guesing, no hiperparameters, no random values
     6) It is apropriate for noise reduction
     
     
More info:
 https://www.researchgate.net/publication/322617800_New_Deep_Learning_Algorithms_beyond_Backpropagation_IBM_Developers_UnConference_2018_Zurich
 https://www.researchgate.net/publication/249011199_Border_Pairs_Method-constructive_MLP_learning_classification_algorithm
 https://www.researchgate.net/publication/263656317_Advances_in_Machine_Learning_Research




This demonstration app is stil under construction!!! 
Learning data is loaded from CSV file. CSV file can contain only numbers.
Each learning pattern is described in one line of the CSV file.
Label is at the end of the each line and must be an integer value.
All other values can be real numbers.

Here is example of logical OR function:  
    
    0,	0,	0
    0,	1,	1
    1,	0,	1
    1,	1,	1

In first two columns is input data, in the third column is class label.


At the moment app finds all border patterns - patterns which are near border lines.
Each border line will be represented with one neuron in the first layer. Border patterns should be separated
so that all patterns are contained in homogenous areas. 
In next layers is repeated the same story. Each homogenous area can be represented in nthe next layer with only one pattern.
Each next layer have fewer neurons. The last layer should be one hot vector.
