####################################################################################
#######     This app is demonstration of BORDER PAIRS METHOD (BPM)
####################################################################################
#
#  This app is stil under construction. BPM is an machine learning classification
#  algortihm for neural networks. BPM have many advantages over the famouse Backpropagation:
#      1) It finds near optimal NN construction
#      2) It is done in one step without iterations
#      3) It finds solution in each atempt
#      4) It uses only useful patterns
#      5) No guesing, no hiperparameters, no random
#      6) and even more ...
#      
#    Learning data is loaded from CSV file. CSV file can contain only numbers.
#    Each learning pattern is described in one line. Label is at the end of
#    line and must be an integer value. Other values can be real numbers 
#    Here is example of logical OR function:
#
#                       0,	0,	0
#                       0,	1,	1
#                       1,	0,	1
#                       1,	1,	1
#
#    In first two columns is input data, in the third column is class label. 
#
#  More info :
#  https://www.researchgate.net/publication/322617800_New_Deep_Learning_Algorithms_beyond_Backpropagation_IBM_Developers_UnConference_2018_Zurich
#  https://www.researchgate.net/publication/249011199_Border_Pairs_Method-constructive_MLP_learning_classification_algorithm
#  https://www.researchgate.net/publication/263656317_Advances_in_Machine_Learning_Research
#
#
###################################################################################


import numpy as np
import matplotlib.pyplot as plt


def loadingData():


	# Read the data from text file
	data = np.genfromtxt('mnist100.csv', delimiter=',')


	# Add a column of serial numbers to the array
	serial_numbers = np.arange(1, data.shape[0] + 1)
	data = np.insert(data, 0, serial_numbers, axis=1)
	patterns,dimensions=data.shape
	dimensions=dimensions-2

	#print(data)
	print("Data is loaded")
	print("Number of learning patterns is: ",patterns)
	print("Dimensionality is: ",dimensions)


	return data


def plotingData(data):
	# Extract the serial numbers, x coordinates, y coordinates, and classes from the data
	serial_numbers = np.int16(data[:, 0])
	x = data[:, 1]
	y = data[:, 2]
	classes = data[:, -1]

	# Create a figure and an axis
	fig, ax = plt.subplots()

	# Plot the points in the scatter plot, using different colors for different classes
	ax.scatter(x[classes == 0], y[classes == 0], color='blue',  label='Class 0')
	ax.scatter(x[classes == 1], y[classes == 1], color='red',   label='Class 1')
	ax.scatter(x[classes  > 1], y[classes  > 1], color='green', label='Class >1')


	# Add the serial numbers to the plot
	for i, txt in enumerate(serial_numbers):
	    ax.text(x[i], y[i], txt)

	# Add a legend to the plot
	ax.legend()

	# Show the plot
	
	print("close graph to continue")
	plt.show()
	

	


def separateData(positivClass):
	positiveData = data[data[:, -1] == positivClass]
	negativeData = data[data[:, -1] != positivClass]
	print("Positive is class: ",positivClass)
	return [positiveData, negativeData]


def borderPatterns(data):
	patterns,dimensions=data.shape
	nearestStranger=np.zeros(patterns)

	# searching nearest patterns for positive data
	#print("----positiveData",positiveData)
	[numberOfPositive,x]=positiveData.shape
	print("Total number of positive patterns: ",numberOfPositive)
	print("Number of currently processed positive pattern:")
	for currentPattern in range(numberOfPositive):
		if (currentPattern%100==0):
			print(currentPattern)
		#print("  ",np.int16(positiveData[currentPattern,0]))
		#print("Value of positive pattern: ",positiveData[currentPattern,1:-1])
		#print("Difference to the negative patterns",negativeData[:,1:-1]-positiveData[currentPattern,1:-1])
		distancesToOpossite=(np.linalg.norm(negativeData[:,1:-1]-positiveData[currentPattern,1:-1],axis=1))
		#print("distances to opossite patterns: ",distancesToOpossite)
		nearestPattern=np.argmin(distancesToOpossite)
		#print("nearest pattern: ",negativeData[ nearestPattern,0])
		nearestStranger[np.int16(positiveData[currentPattern,0])-1]=negativeData[ nearestPattern,0]
	# searching nearest patterns for negative data
	#print("----negativeData",negativeData)
	#print("Positive patterns are finished")
	[numberOfNegative,x]=negativeData.shape
	print("Total number of negative patterns: ",numberOfNegative)
	print("Number of currently processed negative pattern:")
	for currentPattern in range(numberOfNegative):
		if (currentPattern%100==0):
			print(currentPattern)
		#print("  ",np.int16(negativeData[currentPattern,0]))
		#print("Value of negative pattern: ",negativeData[currentPattern,1:-1])
		#print("Difference to the positive patterns",positiveData[:,1:-1]-negativeData[currentPattern,1:-1])
		distancesToOpossite=(np.linalg.norm(positiveData[:,1:-1]-negativeData[currentPattern,1:-1],axis=1))
		#print("distances to opossite patterns: ",distancesToOpossite)
		nearestPattern=np.argmin(distancesToOpossite)
		#print("nearest pattern: ",positiveData[ nearestPattern,0])
		nearestStranger[np.int16(negativeData[currentPattern,0])-1]=positiveData[ nearestPattern,0]
	#print("Negative patterns are finished")


	#print("Nearest stranger (pattern of the opossite class ) from each pattern is:",nearestStranger)
	#print("Nearest strangers are those patterns which define borderline" )



	#counting border patterns
	numberOfBorderPatterns = np.array([0])
	borderPattern = np.empty(patterns, dtype=bool)
	for value in range(1,patterns+1):
		# Create a boolean array that indicates which elements in the vector are equal to the value
		mask = np.equal(nearestStranger, value)

		# Count the number of times the value occurs in the vector
		count = np.count_nonzero(mask)

		if count>0:
			numberOfBorderPatterns=numberOfBorderPatterns+1
			#print("Pattern ",value, " is nearest stranger", count," times.")
			#print(mask)
	print("Number of border patterns: ",numberOfBorderPatterns)
	return nearestStranger


##########################################################
#       MAIN PROGRAM
##########################################################
print("**************************************************")
print("**            Border Pairs Method               **")
print("**                                              **")
print("**   This is an early, non complete version     **")
print("**************************************************")


# loads data from file  
# file name is given in function "loadingData"

print()
print()
data=loadingData()
temp=data.max(axis=0)
numberOfClasses=np.int16(temp[-1])


# drawss patterns
# In case of more than 2 dimensions  it draws first two dimensions
plotingData(data)


for positiveClass in range(numberOfClasses+1):
	#print("******Positiv class is: ", positiveClass)
	[positiveData, negativeData]=separateData(positiveClass)
	print("Border patterns are: ",borderPatterns(data))
