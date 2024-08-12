import numpy as np
from PIL import Image
import numpy.matlib
from matplotlib import pyplot as plt
import sklearn.utils.extmath as alg
from sklearn.neighbors import BallTree

from scipy.sparse.csgraph import floyd_warshall


img = Image.open('indian_noised.png')
img = np.asarray(img, dtype="int32")

# Parameters
nu,rho,ev = 20,5,80

plt.subplot(1, 2, 1)
plt.imshow(img, interpolation='nearest',cmap='gray', vmin=0, vmax=255)
plt.show()
    
def main():       
    print('----Denoising channel----')
    denoi_img = denoise(img, nu, rho, ev) 
    plt.subplot(1, 2, 2)
    plt.imshow(denoi_img, interpolation='nearest',cmap='gray', vmin=0, vmax=255)
    plt.show()
    


def isomap(X, ev, k):
    # Use ball tree to keep nearest neighbor search effecient
    tree = BallTree(X, leaf_size=min(40, 5 * k))
    adj = np.ones([len(X), len(X)]) * float('inf')

    for i in range(len(X)):
        dists, inds = tree.query(X[i].reshape(1, -1), k=k + 1)
        adj[i, inds] = dists

    [d, paths_mat] = floyd_warshall(adj, directed=False, return_predecessors=True, unweighted=False, overwrite=False)

    n = len(paths_mat)
    d = np.zeros([n, n])
    
    print('\tConstructing the graph structure...')
    for i in range(0, len(paths_mat)):
        if i % 500 == 0:
            print('\t' + str(i) + '/' + str(n), end='\t')
        for j in range(i + 1, len(paths_mat)):

            # computes the path
            path = []
            if paths_mat[i, j] == -9999:
                d[i, j] = 0
            else:
                k = j
                path = np.array([j])
                while paths_mat[i, k] != i:
                    path = np.concatenate((path, [paths_mat[i, k]]), axis=0)
                    k = paths_mat[i, k]
                path = np.concatenate((path, [i]), axis=0)
                path = path[::-1]

            d_temp = 0
            for l in range(len(path) - 1):
                d_temp = d_temp + np.linalg.norm(X[path[l], :] - X[path[l + 1], :])

            d[i, j] = d_temp

    # making symetric distance matrices
    d = d + d.T

    # Call MDS
    U, S, Vt = MDS(d, ev)

    return U, S, Vt


def MDS(d, ev):
    # Distances are squared
    d = np.array(d) ** 2
    # Double centerization
    gram = -0.5 * (d - np.mean(d, 0) - np.mean(d, 1).reshape((-1, 1)) + np.mean(d))
    
    # Regular SVD
    # U, S, Vt = np.linalg.svd(gram)
    
    # Randomized SVD
    print('\tUsing randimized SVD...')
    U, S, Vt = alg.randomized_svd(gram, ev)
   
    S_sqrt = np.sqrt(S)    

    return U, S_sqrt, Vt


def denoise(img, nu, rho, ev):
    ##### START: constructing patche matrix and corresponding image addresses
    N = len(img)
    print('Constructing %d patches...' % N ** 2)
    # modify the image to make the biundrary even symetric
    rhoG = int(np.floor(rho / 2))

    lft = np.matlib.repmat(img[:, 0].reshape(N, 1), 1, rhoG)
    # lft = np.matlib.repmat(img[:,:, 0].reshape(N*N, 1), 1, rhoG)
    rgt = np.matlib.repmat(img[:, -1].reshape(N, 1), 1, rhoG)

    imgSym = np.concatenate((lft, img), axis=1)
    imgSym = np.concatenate((imgSym, rgt), axis=1)

    top = np.matlib.repmat(imgSym[0, :], rhoG, 1)
    btm = np.matlib.repmat(imgSym[-1, :], rhoG, 1)

    imgSym = np.concatenate((top, imgSym), axis=0)
    imgSym = np.concatenate((imgSym, btm), axis=0)

    # producing the matrix u (dim is N^2,rho^2) of patches
    u = np.zeros((N ** 2, rho ** 2))
    indU = 0
    for i in range(rhoG, (N + rhoG)):
        for j in range(rhoG, (N + rhoG)):
            #            [i j (i-rhoG) ]
            patch = imgSym[i - rhoG:i + rhoG + 1, j - rhoG:j + rhoG + 1]
            #            patch = patch.T
            u[indU, :] = patch.reshape((1, rho ** 2))
            indU = indU + 1

    # constructing image addresses corresponding to the patches
    x = np.zeros((N ** 2, 2))
    for i in range(N):
        for j in range(N):
            x[N * i + j, :] = [i, j]
    #### END: constructing patches

    ##### START: reducing the dimensionality using Isomap
    print('Using Isomap ...')
    U, S, Vt = isomap(u, ev, nu)

    # producing the matrix uHat (dim is N^2,rho^2) of denoised patches
    uHat = np.zeros((N ** 2, rho ** 2))
    for j in range(rho ** 2):
        val = 0
        for k in range(ev):
            kappa = np.sum(u[:, j] * U[:, k])
            val = val + kappa * U[:, k]
        uHat[:, j] = val
    ##### END:  reducing the dimensionality using Isomap

    ##### START: construction of the noise free image
    print('Constructing the recovered image ...')
    # computing the neighborhoods of x_n's using infinity norm
    xVec = np.concatenate((np.zeros((1, rhoG)), np.ones((1, N)) * range(N), (N - 1) * np.ones((1, rhoG))), axis=1)
    xCord = np.matlib.repmat(xVec.T, 1, N + 2 * rhoG)
    yVec = np.concatenate((np.zeros((1, rhoG)), np.ones((1, N)) * range(N), (N - 1) * np.ones((1, rhoG))), axis=1)
    yCord = np.matlib.repmat(yVec, N + 2 * rhoG, 1)

    infNormX = []
    for i in range(rhoG, N + rhoG):
        infNormX_temp = []
        for j in range(rhoG, N + rhoG):
            xTemp = xCord[i - rhoG:i + rhoG + 1, j - rhoG:j + rhoG + 1]
            # xTemp = np.reshape(xTemp.T,rho**2,1) #order must be str, not int
            xTemp = np.reshape(xTemp.T, rho ** 2)  # order must be one of 'C', 'F', 'A', or 'K' (got '1')
            yTemp = yCord[i - rhoG:i + rhoG + 1, j - rhoG:j + rhoG + 1]
            yTemp = np.reshape(yTemp.T, rho ** 2)
            cordTemp = np.array([xTemp.astype('int'), yTemp.astype('int')]).T
            infNormX_temp.append(cordTemp)

        infNormX.append(infNormX_temp)

    # producing the denoised image
    imgRec = np.zeros((N, N))
    for i in range(N):
        for j in range(N):

            alphaDeno = 0
            for k in range(rho ** 2):
                alphaDeno = alphaDeno + np.exp(-np.linalg.norm([i, j] - infNormX[i][j][k, :]) ** 2)

            pxlRec = 0
            for k in range(rho ** 2):
                xl = infNormX[i][j][k, :]
                alpha = np.exp(-np.linalg.norm([i, j] - xl) ** 2) / alphaDeno
                indInNbhd = rho ** 2 - k - 1
                ul = xl[0] * N + xl[1]
                pxlRec = pxlRec + alpha * uHat[ul, indInNbhd]

            imgRec[i, j] = pxlRec
            #### END: construction of the noise free image
            
    return imgRec
        
main()