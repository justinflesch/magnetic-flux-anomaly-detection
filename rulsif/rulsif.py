import numpy as np

def TimeSeriesDissimilarity(dataset, SampleWidth, RetroWidth, Sigma, Alpha, Lambda):
    # SampleWidth is k in the paper, seen on page 4
    # RetroWidth is n in the paper, seen on page 4
    
    # Get sliding window, with a bit more processing below this will become Y in the paper
    RawSubsequences = np.lib.stride_tricks.sliding_window_view(dataset, SampleWidth, axis=0)
    
    # Subsequences is Y in the paper.
    # The reshape flattens the last 2 dimensions into 1,
    # which the paper seems to say to do in the footnote on page 4
    rawShape = RawSubsequences.shape
    Subsequences = RawSubsequences.reshape((rawShape[0], rawShape[1] *  rawShape[2]))
    
    # I changed this recently, if we start getting unequal shape errors
    dissimilarities = np.zeros(((len(Subsequences) - 2*RetroWidth - 1) // 1 + 1, 3, 2))
    for resIndex, baseIndex in enumerate(range(0, len(Subsequences) - 2*RetroWidth)):
        # Calculate Y1 and Y2 as views of the Subsequences matrix
        # This is fancyY(t) and fancyY(t + n) in the paper
        # Remember RetroWidth is n from the paper
        if resIndex % 1000 == 0:
            print(resIndex, "complete so far\n","baseIndex =",baseIndex)
            
        Y1 = Subsequences[baseIndex : baseIndex + RetroWidth]
        Y2 = Subsequences[baseIndex + RetroWidth : baseIndex + 2 * RetroWidth]
        total, forward, backward = BidirectionalDissimilarity(Y1, Y2, Sigma, Alpha, Lambda)
        
        pointX = baseIndex + RetroWidth + SampleWidth // 2
        dissimilarities[resIndex] = ((pointX, total), (pointX, forward), (pointX, backward))
    
    return dissimilarities

def BidirectionalDissimilarity(Y1, Y2, Sigma, Alpha, Lambda):
    if Y1.shape != Y2.shape:
        raise ValueError("Y1.shape != Y2.shape")
    forward = CalculateDissimilarity(Y1, Y2, Sigma, Alpha, Lambda)
    backward = CalculateDissimilarity(Y2, Y1, Sigma, Alpha, Lambda)
    return (forward + backward, forward, backward)

def CalculateDissimilarity(Y1, Y2, Sigma, Alpha, Lambda):
    # First, find the parameters theta
    Thetas = FindThetas(Y1, Y2, Sigma, Alpha, Lambda)
    
    # Now calculate the relative divergence
    Divergence = RelativeDivergence(Y1, Y2, Thetas, Sigma, Alpha)
    
    return Divergence

def FindThetas(Y1, Y2, Sigma, Alpha, Lambda):
    # Equation (8), from page 9

    h = hVector(Y1, Sigma, Lambda)
    H = HMatrix(Y1, Y2, Sigma, Alpha)
    Identity = np.identity(Y1.shape[0])
    # @ is matrix multiplication (In Numpy for Python 3.5+)
    Inner = H + Lambda * Identity
    Thetas = np.linalg.inv(Inner) @ h
    
    return Thetas

def RelativeDivergence(Y1, Y2, Thetas, Sigma, Alpha):
    # Equation from page 11
    
    Divergence = 0
    width = Y1.shape[0]
    
    # Legacy, new logic should have same result but be much faster hopefully
    #sum = 0
    #for i in range(width):
    #    ratio = EstimateRatio(Y1[i], Y1, Thetas, Sigma)
    #    sum += np.square(ratio)
    
    matrix = GaussMatrixKernel(Y1, Y1, Sigma) * Thetas
    result = np.square(matrix.sum(axis=1)).sum(axis=0)
    
    Divergence += -Alpha / (2 * width) * result
    
    # Legacy good implementation
    #sum = 0
    #for i in range(width):
    #    ratio = EstimateRatio(Y2[i], Y1, Thetas, Sigma)
    #    sum += np.square(ratio)
    
    matrix = GaussMatrixKernel(Y2, Y1, Sigma) * Thetas
    result = np.square(matrix.sum(axis=1)).sum(axis=0)
    
    Divergence += -(1 - Alpha) / (2 * width) * result
    
    # Legacy good implementation
    #sum = 0
    #for i in range(width):
    #    ratio = EstimateRatio(Y1[i], Y1, Thetas, Sigma)
    #    sum += ratio
        
    matrix = GaussMatrixKernel(Y1, Y1, Sigma) * Thetas
    result = matrix.sum()
    
    Divergence += 1 / width * result
    
    Divergence += -1/2
    
    return Divergence

# This function is now unused; all logic was rolled into RelativeDivergence
def EstimateRatio(Y1, Y2, Thetas, Sigma):
    # Equation from page 7
    # Y1 is single Y, Y2 is entire fancyY
    # Should either vectorize this, or move logic into RelativeDivergence and vectorize there
    
    deltas = Y2 - Y1
    
    squares = (deltas * deltas).sum(axis=1)
    arguments = -1.0/(2*Sigma*Sigma) * squares
    gaussResults = np.exp(arguments)
    
    result = np.dot(Thetas, gaussResults)
    
    #sum = 0
    #for i in range(Y2.shape[0]):
    #    sum += Thetas[i] * GaussKernel(Y1, Y2[i], Sigma)
    
    return result # / Y2.shape[0]

def hVector(Y1, Sigma, Lambda):
    # Equation is on page 9
    
    SumMatrix = GaussMatrixKernel(Y1, Y1, Sigma)
    SumVector = np.sum(SumMatrix, axis = 0) # If this is the wrong axis tranpose the output of GuassMatrixKernel (dont transpose here)
    Result = 1/Y1.shape[0] * SumVector
    
    return Result

def HMatrix(Y1, Y2, Sigma, Alpha):
    # Equation from page 11
    
    AllKOfY = GaussMatrixKernel(Y1, Y1, Sigma)
    
    # So K(y_i, y_j) = AllKOfY[i,j]
    shape = Y1.shape
    width = shape[0]
    H = np.zeros((width, width))
    H1 = np.zeros((width, width))
    H2 = np.zeros((width, width))
    # We can start with the naive version
    
    # Legacy slow code
    #for x in range(0, width):
    #    for y in range(0, width):
    #        sum = 0
    #        for i in range(0, width):
    #            sum += AllKOfY[i,x] * AllKOfY[i,y]
    #        H1[x,y] += sum
    #
    #H1 = H1 * (Alpha / width)
    
    for i in range(0, width):
        partMatrix = np.outer(AllKOfY[i, :], AllKOfY[i, :])
        H1 = H1 + partMatrix
    H1 = H1 * (Alpha / width)
    
    AllKOfYPrime = GaussMatrixKernel(Y2, Y1, Sigma)
    
    # Legacy unvectorized code
    #for x in range(0, width):
    #    for y in range(0, width):
    #        sum = 0
    #        for i in range(0, width):
    #            sum += AllKOfYPrime[i,x] * AllKOfYPrime[i,y]
    #        H2[x,y] += sum
    #H2 = H2 * ((1 - Alpha) / width)
    
    for i in range(0, width):
        partMatrix = np.outer(AllKOfYPrime[i, :], AllKOfYPrime[i, :])
        H2 = H2 + partMatrix
    H2 = H2 * ((1 - Alpha) / width)
    
    H = H1 + H2
    
    return H

def GaussMatrixKernel(X1, X2, Sigma):
    # Returns a matrix of gaussian kernel results
    # Equation from page 7, but on an entire matrix instead of one sample
    
    Deltas = vectorizedMatrixSubtract(X1, X2)
    Squares = (Deltas * Deltas).sum(axis=2)
    Argument = -1.0/(2*Sigma*Sigma) * Squares
    #print(Argument)
    Result = np.exp(Argument)
    
    return Result

def vectorizedMatrixSubtract(X1, X2):
    # Evil broadcasting magic
    
    Result = X1[:, None, :] - X2[None, :, :]
    return Result

# This is bad because it isn't vectorized; there HAS to be a better way
# See above function for the better way; this is kept for comparison
def badSpecialMatrixSubtract(X1, X2):
    result = np.zeros((X1.shape[0], X1.shape[0], X1.shape[1]))
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            result[i,j] = X1[i] - X2[j]
    
    return result