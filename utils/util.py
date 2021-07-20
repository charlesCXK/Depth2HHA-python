# --*-- coding:utf-8 --*--
import numpy as np
import cv2
from scipy import signal

'''
helper function
'''
def filterItChopOff(f, r, sp):
    f[np.isnan(f)] = 0
    H, W, d = f.shape
    B = np.ones([2 * r + 1, 2 * r + 1])     # 2r+1 * 2r+1 neighbourhood

    minSP = cv2.erode(sp, B, iterations=1)
    maxSP = cv2.dilate(sp, B, iterations=1)

    ind = np.where(np.logical_or(minSP != sp, maxSP != sp))

    spInd = np.reshape(range(np.size(sp)), sp.shape,'F')

    delta = np.zeros(f.shape)
    delta = np.reshape(delta, (H * W, d), 'F')
    f = np.reshape(f, (H * W, d),'F')

    # calculate delta

    I, J = np.unravel_index(ind, [H, W], 'C')
    for i in range(np.size(ind)):
        x = I[i]
        y = J[i]
        clipInd = spInd[max(0, x - r):min(H-1, x + r), max(0, y - r):min(W-1, y + r)]
        diffInd = clipInd[sp[clipInd] != sp[x, y]]
        delta[ind[i], :] = np.sum(f[diffInd, :], 1)
    delta = np.reshape(delta, (H, W, d), 'F')
    f = np.reshape(f, (H, W, d), 'F')
    fFilt = np.zeros([H, W, d])

    for i in range(f.shape[2]):
        #  fFilt(:,:,i) = filter2(B, f(:,:,i));
        tmp = cv2.filter2D(np.rot90(f[:, :, i], 2), -1, np.rot90(np.rot90(B, 2), 2))
        tmp = signal.convolve2d(np.rot90(f[:, :, i], 2), np.rot90(np.rot90(B, 2), 2), mode="same")
        fFilt[:, :, i] = np.rot90(tmp, 2)
    fFilt = fFilt - delta
    return fFilt

'''
helper function
'''
def mutiplyIt(AtA_1, Atb):
    result = np.zeros([Atb.shape[0], Atb.shape[1], 3])
    result[:, :, 0] = np.multiply(AtA_1[:, :, 0], Atb[:, :, 0]) + np.multiply(AtA_1[:, :, 1],
                                                                              Atb[:, :, 1]) + np.multiply(
        AtA_1[:, :, 2], Atb[:, :, 2])
    result[:, :, 1] = np.multiply(AtA_1[:, :, 1], Atb[:, :, 0]) + np.multiply(AtA_1[:, :, 3],
                                                                              Atb[:, :, 1]) + np.multiply(
        AtA_1[:, :, 4], Atb[:, :, 2])
    result[:, :, 2] = np.multiply(AtA_1[:, :, 2], Atb[:, :, 0]) + np.multiply(AtA_1[:, :, 4],
                                                                              Atb[:, :, 1]) + np.multiply(
        AtA_1[:, :, 5], Atb[:, :, 2])
    return result

'''
helper function
'''
def invertIt(AtA):
    AtA_1 = np.zeros([AtA.shape[0], AtA.shape[1], 6])
    AtA_1[:, :, 0] = np.multiply(AtA[:, :, 3], AtA[:, :, 5]) - np.multiply(AtA[:, :, 4], AtA[:, :, 4])
    AtA_1[:, :, 1] = -np.multiply(AtA[:, :, 1], AtA[:, :, 5]) + np.multiply(AtA[:, :, 2], AtA[:, :, 4])
    AtA_1[:, :, 2] = np.multiply(AtA[:, :, 1], AtA[:, :, 4]) - np.multiply(AtA[:, :, 2], AtA[:, :, 3])
    AtA_1[:, :, 3] = np.multiply(AtA[:, :, 0], AtA[:, :, 5]) - np.multiply(AtA[:, :, 2], AtA[:, :, 2])
    AtA_1[:, :, 4] = -np.multiply(AtA[:, :, 0], AtA[:, :, 4]) + np.multiply(AtA[:, :, 1], AtA[:, :, 2])
    AtA_1[:, :, 5] = np.multiply(AtA[:, :, 0], AtA[:, :, 3]) - np.multiply(AtA[:, :, 1], AtA[:, :, 1])

    x1 = np.multiply(AtA[:, :, 0], AtA_1[:, :, 0])
    x2 = np.multiply(AtA[:, :, 1], AtA_1[:, :, 1])
    x3 = np.multiply(AtA[:, :, 2], AtA_1[:, :, 2])

    detAta = x1 + x2 + x3
    return AtA_1, detAta

'''
Compute the direction of gravity
N: normal field
iter: number of 'big' iterations
'''
def getYDir(N, angleThresh, iter, y0):
    y = y0
    for i in range(len(angleThresh)):
        thresh = np.pi * angleThresh[i] / 180   # convert it to radian measure
        y = getYDirHelper(N, y, thresh, iter[i])
    return y

'''
N: HxWx3 matrix with normal at each pixel.
y0: the initial gravity direction
thresh: in degrees the threshold for mapping to parallel to gravity and perpendicular to gravity
iter: number of iterations to perform
'''
def getYDirHelper(N, y0, thresh, num_iter):
    dim = N.shape[0] * N.shape[1]

    # change the third dimension to the first-order. (480, 680, 3) => (3, 480, 680)
    nn = np.swapaxes(np.swapaxes(N,0,2),1,2)
    nn = np.reshape(nn, (3, dim), 'F')

    # remove these whose number is NAN
    idx = np.where(np.invert(np.isnan(nn[0,:])))[0]
    nn = nn[:,idx]

    # Set it up as a optimization problem
    yDir = y0;
    for i in range(num_iter):
        sim0 = np.dot(yDir.T, nn)
        indF = abs(sim0) > np.cos(thresh)       # calculate 'floor' set.    |sin(theta)| < sin(thresh) ==> |cos(theta)| > cos(thresh)
        indW = abs(sim0) < np.sin(thresh)       # calculate 'wall' set.
        if(len(indF.shape) == 2):
            NF = nn[:, indF[0,:]]
            NW = nn[:, indW[0,:]]
        else:
            NF = nn[:, indF]
            NW = nn[:, indW]
        A = np.dot(NW, NW.T) - np.dot(NF, NF.T)
        b = np.zeros([3,1])
        c = NF.shape[1]
        w,v = np.linalg.eig(A)      # w:eigenvalues; v:eigenvectors
        min_ind = np.argmin(w)      # min index
        newYDir = v[:,min_ind]
        yDir = newYDir * np.sign(np.dot(yDir.T, newYDir))
    return yDir

'''
getRMatrix: Generate a rotation matrix that
            if yf is a scalar, rotates about axis yi by yf degrees
            if yf is an axis, rotates yi to yf in the direction given by yi x yf
Input: yi is an axis 3x1 vector
       yf could be a scalar of axis

'''
def getRMatrix(yi, yf):
    if (np.isscalar(yf)):
        ax = yi / np.linalg.norm(yi)        # norm(A) = max(svd(A))
        phi = yf
    else:
        yi = yi / np.linalg.norm(yi)
        yf = yf / np.linalg.norm(yf)
        ax = np.cross(yi.T, yf.T).T
        ax = ax / np.linalg.norm(ax)
        # find angle of rotation
        phi = np.degrees(np.arccos(np.dot(yi.T, yf)))

    if (abs(phi) > 0.1):
        phi = phi * (np.pi / 180)

        s_hat = np.array([[0, -ax[2], ax[1]],
                          [ax[2], 0, -ax[0]],
                          [-ax[1], ax[0], 0]])
        R = np.eye(3) + np.sin(phi) * s_hat + (1 - np.cos(phi)) * np.dot(s_hat, s_hat)      # dot???
    else:
        R = np.eye(3)
    return R

'''
Calibration of gravity direction 
'''
def rotatePC(pc, R):
    if(np.array_equal(R, np.eye(3))):
        return pc
    else:
        R = R.astype(np.float64)
        dim = pc.shape[0] * pc.shape[1]
        pc = np.swapaxes(np.swapaxes(pc, 0, 2), 1, 2)
        res = np.reshape(pc, (3, dim), 'F')
        res = np.dot(R, res)
        res = np.reshape(res, pc.shape, 'F')
        res = np.swapaxes(np.swapaxes(res, 0, 1), 1, 2)
        return res