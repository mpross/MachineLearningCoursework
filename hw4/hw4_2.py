import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

def generateData(n):

    i = np.array(range(1, n+1))
    x = (i-1)/(n-1)

    y = np.zeros(n)
    for i in range(1, 5):
        y += 10*(np.array(x > i/5, dtype="float"))

    f = np.copy(y)

    y += np.random.randn(n)

    y[25] = 0

    return x, y, f

def kernel(x, z, gam):

    K = np.zeros((x.size, z.size))

    for i in range(0, x.size):
        for j in range(0, z.size):
            K[i, j] = np.exp(-gam*(x[i]-z[j])**2)

    return K


def calcD(n):

    D = np.zeros((n, n-1))

    for i in range(0, n):
        for j in range(0, n-1):
            if i == j:
                D[i, j] = 1

            if i == j-1:
                D[i, j] = -1
    return D.T


def leave1(x, y):


    lamb = cp.Parameter(nonneg=True)
    # lamb2 = cp.Parameter(nonneg=True)
    lamb2Best=0
    # lambList = np.array(range(10, 20))/100
    # gamList = np.array(range(40, 100, 10))
    lambList = np.random.uniform(1, 10, 10)/100
    lamb2List = np.random.uniform(1, 10, 1)/100
    gamList = np.random.uniform(150, 250, 10)

    lambList.sort()
    gamList.sort()

    errList = np.zeros((lambList.size, gamList.size))
    err2List = np.zeros((lambList.size, lamb2List.size))

    errBest=10**100
    for gamIndex in range(gamList.size):
        print(gamIndex)
        for lambIndex in range(lambList.size):
            # for lamb2Index in range(lamb2List.size):
            errMean=0
            for i in range(0, y.size-1):
                lamb.value = lambList[lambIndex]
                # lamb2.value = lamb2List[lamb2Index]

                K = kernel(np.delete(x, i), np.delete(x, i), gamList[gamIndex])
                D = calcD(K.shape[1])

                a = cp.Variable(K.shape[1])

                err = cp.sum_squares(y[i] - np.sum(K[:, i] * a))

                # objective = cp.Minimize(cp.sum_squares(np.delete(y, i) - np.sum(K * a)) + lamb * cp.quad_form(a, K))
                # objective = cp.Minimize(cp.sum(cp.huber(np.delete(y, i) - np.sum(K * a))) + lamb * cp.quad_form(a, K))
                # objective = cp.Minimize(
                #     cp.sum_squares(np.delete(y, i) - np.sum(K * a)) +
                #     lamb * cp.quad_form(a, K) + lamb2 * cp.norm1(D * (K * a)))
                objective = cp.Minimize(cp.sum_squares(np.delete(y, i) - np.sum(K * a)) + lamb * cp.quad_form(a, K))
                constraints = [cp.matmul(cp.matmul(D, K), a) >= 0]

                # prob = cp.Problem(objective)
                prob = cp.Problem(objective, constraints)

                prob.solve()

                errMean += err.value

            if errMean < errBest:
                lambBest = lambList[lambIndex]
                # lamb2Best = lambList[lamb2Index]
                gamBest = gamList[gamIndex]
                errBest = errMean

            errList[lambIndex, gamIndex] = errMean
            # err2List[lambIndex, lamb2Index] = errMean



    plt.figure()
    plt.imshow(errList)
    plt.ylabel('lambda')
    plt.xlabel('gamma')
    plt.draw()

    plt.figure()
    plt.imshow(err2List)
    plt.ylabel('lambda2')
    plt.xlabel('lambda1')
    plt.draw()

    return lambBest, lamb2Best, gamBest

n = 50

x, y, f = generateData(n)

D = calcD(n)

lamb, lamb2, gam = leave1(x, y)

K = kernel(x, x, gam)

a = cp.Variable(K.shape[1])

err = cp.sum_squares(y - np.sum(K * a))

# objective = cp.Minimize(cp.sum_squares(y - np.sum(K * a)) + lamb * cp.quad_form(a, K))
# objective = cp.Minimize(cp.sum(cp.huber(y - np.sum(K * a))) + lamb * cp.quad_form(a, K))
# objective = cp.Minimize(
#                     cp.sum_squares(y - np.sum(K * a)) +
#                     lamb * cp.quad_form(a, K) + lamb2 * cp.norm1(D * (K * a)))
objective = cp.Minimize(cp.sum_squares(y - np.sum(K * a)) + lamb * cp.quad_form(a, K))
constraints = [cp.matmul(cp.matmul(D, K), a)>= 0]

# prob = cp.Problem(objective)
prob = cp.Problem(objective, constraints)

prob.solve()

plt.figure()
plt.plot(x, y, '.')
plt.plot(x, f)
plt.plot(x, np.dot(K, a.value))
plt.legend(('Original Data', 'True', 'Estimate'))
plt.title('Quad '+' lambda: '+str(round(lamb,2))+' gamma: '+str(round(gam,2)))#lambda1: '+str(round(lamb2,2))
plt.draw()
plt.savefig('2_quad.pdf')


print('Lambda '+str(lamb))
print('Gamma '+str(gam))
plt.show()
