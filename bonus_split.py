import sklearn as skl
def bonus_split(data, label, train_size):
  num_rows, num_cols = data.shape
  new_train = np.zeros(shape = (int (train_size * num_rows), num_cols))
  new_test = np.zeros(shape = (int ((1 - train_size) * num_rows), num_cols))
  index_train = 0
  index_test = 0
  train_label = np.zeros(int (train_size * num_rows))
  test_label = np.zeros(int ((1 - train_size) * num_rows))
  for i in range (0, len(data), 10):
    for j in range(i, i + 7):
      train_label[index_train] = label[j] 
      new_train[index_train] = data[j]
      index_train = index_train + 1
    for j in range(i + 7, i + 10):
      test_label[index_test] = label[j] 
      new_test[index_test] = data[j]
      index_test = index_test + 1
  return new_train, new_test, train_label, test_label
train_data, test_data, train_labels, test_labels = bonus_split(data, labels, 0.7)

train_data_transformed, test_data_transformed = lda_sklearn()

# KNN
accuracy = knnClassifier(5, train_data_transformed, test_data_transformed)  

print('> accuracy is', accuracy * 100, '%')


# TESTNG RESULTS
train_data_transformed, test_data_transformed = pca_sklearn(0.85)
# KNN
accuracy = knnClassifier(5, train_data_transformed, test_data_transformed)  
print('> accuracy is', accuracy * 100, '%')
