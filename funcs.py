import numpy as np

from cv2 import SIFT_create, BFMatcher, norm 
from numba import njit


# Preprocessing

def generateRandom(src_Pts, dest_Pts, N):
    # Randomly selects N coresponding points
    
    r = np.random.choice(len(src_Pts), N)
    src = [src_Pts[i] for i in r]
    dest = [dest_Pts[i] for i in r]
    return np.asarray(src, dtype=np.float32), np.asarray(dest, dtype=np.float32)
    
    
def ransacHomography(src_Pts, dst_Pts):
    
    maxI = 0
    maxLSrc = []
    maxLDest = []
    
    for i in range(70):
        srcP, destP = generateRandom(src_Pts, dst_Pts, 4)
        H = find_H(srcP, destP, 4)
        inlines = 0
        linesSrc = []
        lineDest = []
        for p1, p2 in zip(src_Pts, dst_Pts):
            p1U = (np.append(p1, 1)).reshape(3, 1)
            p2e = H.dot(p1U)
            p2e = (p2e / p2e[2])[:2].reshape(1, 2)[0]
            if norm(p2 - p2e) < 10:
                inlines += 1
                linesSrc.append(p1)
                lineDest.append(p2)
        if inlines > maxI:
            maxI = inlines
            maxLSrc = linesSrc.copy()
            maxLSrc = np.asarray(maxLSrc, dtype=np.float32)
            maxLDest = lineDest.copy()
            maxLDest = np.asarray(maxLDest, dtype=np.float32)
    Hf = find_H(maxLSrc, maxLDest, maxI)
    return Hf
    
    
def find_H(src, dest, N):
    
    A = []
    for i in range(N):
        x, y = src[i][0], src[i][1]
        xp, yp = dest[i][0], dest[i][1]
        A.append([x, y, 1, 0, 0, 0, -x * xp, -xp * y, -xp])
        A.append([0, 0, 0, x, y, 1, -yp * x, -yp * y, -yp])
        
    A = np.asarray(A)
    U, S, Vh = np.linalg.svd(A)
    L = Vh[-1, :] / Vh[-1, -1]
    H = L.reshape(3, 3)
    return H
    
    
def get_H(img1, img2):
    '''
    img1 - center
    img2 - panned
    '''
    
    # detect kepoints and their descriptor for 'img1' using SIFT
    sift = SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # Keypoints matching
    bf = BFMatcher() 
    matches = bf.knnMatch(des1,des2, k=2)
    good = []

    for m in matches:
        if (m[0].distance < 0.5*m[1].distance):
            good.append(m)
    matches = np.asarray(good)

    src = np.float32([ kp1[m.queryIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
    dst = np.float32([ kp2[m.trainIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
    
    
    # Matrix
    H = ransacHomography(src[:200, 0, :], dst[:200, 0, :])
    
    return H
    
    
@njit
def reconstruction(img2_clr, H, y_max, y_min, x_max, x_min, img1_shape ):

    img2_warped = np.zeros( (y_max - y_min, x_max - x_min, 3) ) 

    for y in range(0 + y_min, img2_warped.shape[0]):
        for x in range(0 + x_min, img2_warped.shape[1]):
            
            p_src = np.array([x, y, 1], dtype = np.float64)
            p_dst = H.dot(p_src)
            p_dst[0] = p_dst[0]/p_dst[2]
            p_dst[1] = p_dst[1]/p_dst[2]
            p_dst[2] = p_dst[2]/p_dst[2]

            x_new = int(p_dst[0])
            y_new = int(p_dst[1])

            if 0 < y_new < img1_shape[0]-1 and 0 < x_new < img1_shape[1]-1:
                img2_warped[y - y_min, x - x_min] = ( (x_new - int(x_new))*(y_new - int(y_new))*img2_clr[int(y_new+1), int(x_new+1)] 

                                                        +(int(x_new+1) - x_new)*(y_new - int(y_new))*img2_clr[int(y_new+1), int(x_new)]

                                                        +(x_new - int(x_new))*(int(y_new+1) - y_new)*img2_clr[int(y_new), int(x_new+1)]

                                                        +(int(x_new+1) - x_new)*(int(y_new+1) - y_new)*img2_clr[int(y_new), int(x_new)] )
                    
                    
    return img2_warped
    
    
def warped_points(Hinv, s):
    t = [0,0,0,0]
    t[0] = np.dot(Hinv, s[0])
    t[1] = np.dot(Hinv, s[1])
    t[2] = np.dot(Hinv, s[2])
    t[3] = np.dot(Hinv, s[3])
    
    t[0] = t[0]/t[0][2]
    t[1] = t[1]/t[1][2]
    t[2] = t[2]/t[2][2]
    t[3] = t[3]/t[3][2]
    
    return t
    
    
# TRWS

@njit
def initN(h,w):
    
    N = np.full((h*w, 4), -1, dtype = np.int32)

    for i in range(h):
        for j in range(w):

            index = j + i*w

            # left
            if j > 0:
                left_index = j - 1 + i*w
                N[index, 0] = left_index
            # right
            if j < w - 1:
                right_index = j + 1 + i*w
                N[index, 1] = right_index
             # up
            if i > 0:
                up_index = j + (i-1)*w
                N[index, 2] = up_index
            # down
            if i < h - 1:
                down_index = j + (i+1)*w
                N[index, 3] = down_index
                
    return N
    
    



@njit
def init_q(IMGS, C, N, w, h):
    q = np.zeros((w*h, len(C)), dtype=np.float32)
    
    zero = np.zeros(3, dtype=np.uint8)
    for i in range(w*h):
        y,x = i // w, i % w
            
        if (IMGS[0][y,x,:] + IMGS[1][y,x,:] + IMGS[2][y,x,:] == np.zeros(3, dtype=np.uint8)).all():
            q[i, 0] = 0
            q[i, 1] = 0
            q[i, 2] = 0
            
        else:
            for c in C:
                if (IMGS[c][y,x,:] == zero).all():
                    q[i, c] = np.inf
                else:
                    q[i, c] = 0   
    return q


@njit(fastmath=True)
def norm(l):
    s = 0.
    for i in range(l.shape[0]):
        s += l[i]**2
    return np.sqrt(s)


@njit
def init_g(IMGS, C, N, w, h):
    
    g = np.zeros((w*h, 4, len(C), len(C)), dtype=np.float32)
    
    for i in range(w*h):
        y,x = i // w, i % w
        
        for n_ind, n in enumerate(N[i]):
            y_, x_ = n // w, n % w
            
            for c in C:
                for c_ in C:
                    g[i, n_ind, c, c_] = norm(IMGS[c][y,x,:] - IMGS[c_][y,x,:]) + norm(IMGS[c][y_,x_,:] - IMGS[c_][y_,x_,:])    
    return g
    
    




@njit
def initDR(D, R, C, N, q, g, w, h):
    
    for y in range(h-1, -1, -1):
        for x in range(w-1, -1, -1):
            for c in C:
                index = x + w*y
                
                # D
                if y == h-1:
                    D[y,x,c] = 0
                else:
                    index2 = N[index, 3]
                    
                    best_score = np.inf
                    for c_ in C:
                        foo = D[y+1, x, c_] + 0.5*q[index2, c_] + g[index, 3, c, c_]
                        if foo < best_score:
                            best_score = foo
                    D[y,x,c] = best_score
                    
                # R
                if x == w-1:
                    R[y,x,c] = 0
                else:
                    index2 = N[index, 1]
                    best_score = np.inf
                    for c_ in C:
                        foo = R[y, x+1, c_] + 0.5*q[index2, c_] + g[index, 1, c, c_]
                        if foo < best_score:
                            best_score = foo
                    R[y,x,c] = best_score

    return D, R
    
    
@njit
def iteration(L, U, R, D, Phi, N, q, g, w, h, C):
    
    # Forward pass
    for y in range(0, h):
        for x in range(0, w):
            for c in C:
                index = x + w*y
                
                # U
                index2 = N[index, 2]
                if y == 0:
                    U[y,x,c] = 0
                else:
                    best_score = np.inf
                    for c_ in C:
                        foo = U[y-1,x,c_] + 0.5*q[index2, c_] + g[index, 2, c, c_] + Phi[index2, c_]
                        if foo < best_score:
                            best_score = foo
                    U[y,x,c] = best_score   
                # L
                index2 = N[index, 0]
                if x == 0:
                    L[y,x,c] = 0
                else:
                    best_score = np.inf
                    for c_ in C:
                        foo = L[y,x-1,c_] + 0.5*q[index2, c_] + g[index, 0, c, c_] - Phi[index2, c_]
                        if foo < best_score:
                            best_score = foo
                    L[y,x,c] = best_score
                # Phi
                Phi[index, c] = ( L[y,x,c] + R[y,x,c] - U[y,x,c] - D[y,x,c] )*0.5
                
         
    # Backward pass
    for y in range(h-1, -1, -1):
        for x in range(w-1, -1, -1):
            for c in C:
                index = x + w*y
                
                # D
                if y == h-1:                   
                    D[y,x,c] = 0
                else:
                    index2 = N[index, 3]
                    
                    best_score = np.inf
                    for c_ in C:
                        foo = D[y+1, x, c_] + 0.5*q[index2, c_] + g[index, 3, c, c_] + Phi[index2, c_]
                        if foo < best_score:
                            best_score = foo
                    D[y,x,c] = best_score
                    
                # R
                if x == w-1:            
                    R[y,x,c] = 0
                else:
                    index2 = N[index, 1]
                    best_score = np.inf
                    for c_ in C:
                        foo = R[y, x+1, c_] + 0.5*q[index2, c_] + g[index, 1, c, c_] - Phi[index2, c_]
                        if foo < best_score:
                            best_score = foo
                    R[y,x,c] = best_score
                    
                # Phi
                Phi[index, c] = ( L[y,x,c] + R[y,x,c] - U[y,x,c] - D[y,x,c] )*0.5
                
    return L, U, R, D, Phi
    
    


@njit
def reconstruct(L, R, Phi, C, q, w, h):
    Res = np.zeros((h,w), dtype=np.uint8)
    
    for y in range(0, h):
        for x in range(0, w):
            for c in C:
                index = x + w*y

                best_score = np.inf
                best_c = -1
                for c_ in C:
                    foo = L[y,x,c_] + R[y,x,c_] + 0.5*q[index, c_] - Phi[index, c_]
                    if foo < best_score:
                            best_score = foo
                            best_c = c_

                Res[y,x] = best_c
    return Res
    
    
@njit
def render_result(Res, IMGS, w, h):
    
    res_img = np.zeros_like(IMGS[0])
    
    for y in range(0, h):
        for x in range(0,w):
            
            res_img[y,x,:] = IMGS[Res[y,x]][y,x,:]
            
    return res_img


