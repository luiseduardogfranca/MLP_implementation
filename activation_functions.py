def sig(x, deriv=False):
	if deriv == True:
		return sig(x)*(1-sig(x))

	return 1/(1+np.exp(-x))

def ReLU(x, deriv=False):
	if deriv:
		return (x>0)+(x<=0)
	return np.maximum(0,x)

def LeakyReLU(x, deriv=False):
	if deriv == True:
		return 1. * (x >= 0) + (0.01*(x<0))
	return ((x < 0 )*(0.01*x)) + ((x >= 0)*x)

def tanh(x, deriv=False):
	if deriv == True:
		return (1-(x**2))

	return np.tanh(x)
