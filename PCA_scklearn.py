from sklearn.decomposition import PCA

# #############################################################
# pca using sklearn built in functions
# input:
#   alpha: pca variance threshold
# output:
#   train_data_transformed: train_data after applying pca
#   test_data_transformed: test_data after applying pca
# #############################################################
def pca_sklearn(alpha=0.85):
  # create the PCA instance
  pca = PCA(n_components=alpha, svd_solver='full')
  # fit on data
  pca.fit(train_data)
  # transform data
  train_data_transformed = pca.transform(train_data)
  test_data_transformed = pca.transform(test_data)
  return train_data_transformed, test_data_transformed


