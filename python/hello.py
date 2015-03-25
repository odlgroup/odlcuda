from SimRec2DPy import utils as SR
import numpy as np

#Hello world
print SR.greet()

a = SR.EigenVector([1,2,3])
b = SR.EigenVector([4,5,3])
c = a+b+b
print(c)


d = np.array([1,2,3])
print(SR.printArray([1,2,3]))
