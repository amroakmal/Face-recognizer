# #############################################################
# pca using the pseudo code
# input:
#   eigenvalues: eigen values of the covariance matrix
#   eigenvectors: eigen vectors of the covariance matrix
#   alpha: pca variance threshold
# output:
#   train_data_transformed: train_data after applying pca
#   test_data_transformed: test_data after applying pca
# #############################################################
def pca_(eigenvalues, eigenvectors, alpha=0.85):
  eig_vals = eigenvalues[::-1]
  eig_vecs = eigenvectors[::-1]
  
  # deciding the appropriate dimensions
  eigenvalues_sum = sum(eig_vals)
  
  for n_components in range(len(eig_vals)):
    eig_vals_ = eig_vals[:n_components]
    if (sum(eig_vals_)/eigenvalues_sum) >= alpha:
      break

  # projecting data on the new space
  v = eig_vecs[::-1].T[::-1]
  train_data_transformed = np.dot(train_data, v[:n_components].T)
  test_data_transformed = np.dot(test_data, v[:n_components].T)
  
  return train_data_transformed, test_data_transformed
