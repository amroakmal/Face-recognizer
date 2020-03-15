def LDA_PseudoCode(data, test, nk):
  num_rows, num_cols = data.shape
  mean_vector = np.zeros(shape = (int (num_rows / nk), num_cols))
  index = 0
  for i in range(0, len(data), nk):
    x = data[i : i + nk, :]
    y = np.mean(x, axis = 0)
    mean_vector[index] = np.transpose(y)
    index = index + 1
  # mean_vector: (40, 10304) : mean_vector[i] : all mean of classes for person i
  dimen = num_cols
  Sb = np.zeros(shape = (dimen, dimen))
  overallSampleMean = np.mean(data, axis = 0)
  for i in range(0, 40):
    currentSample = np.outer((mean_vector[i] - overallSampleMean), (mean_vector[i] - overallSampleMean))
    # currentSample = np.outer((mean_vector[i] - overallSampleMean), np.transpose(mean_vector[i] - overallSampleMean))
    # currentSample = (mean_vector[i] - overallSampleMean).T.dot(mean_vector[i] - overallSampleMean)
    currentSample = currentSample * nk
    Sb = Sb + currentSample
  
  Z = np.zeros(shape = (num_rows, num_cols))
  index = 0
  for i in range(0, len(data), nk):
    for j in range(i, i + nk):
      x = data[j] - mean_vector[index]
      Z[j] = x
    index = index + 1
  S = np.zeros(shape = (num_cols, num_cols))
  for i in range(0, 40):
    auxMatr = np.zeros(shape = (num_cols, num_cols))
    for j in range(i * nk, i * nk + nk):
      auxMatr = auxMatr + np.outer(Z[j].T, Z[j])
    S = S + auxMatr

  invS = np.linalg.pinv(S, hermitian = True)
  fin = invS.dot(Sb)
  eigenValues, eigenVectors = np.linalg.eig(fin)
  idx = eigenValues.argsort()[::-1]   

  eigenValues = eigenValues[idx]
  eigenVectors = eigenVectors[:,idx]

  project_training = data.dot(eigenVectors[:, 0: 39].real)
  project_test = test.dot(eigenVectors[:, 0: 39].real)
  
  return project_training, project_test


train_data_transformed, test_data_transformed = LDA_PseudoCode(train_data, test_data, 5)
k = best_accuracy_k(k_list, train_data_transformed, test_data_transformed)
print('> LDA best k is', k)

