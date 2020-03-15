from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# #############################################################
# lda using sklearn built in functions
# output:
#   train_data_transformed: train_data after applying lda
#   test_data_transformed: test_data after applying lda
# #############################################################
def lda_sklearn():
  # create the LDA instance
  lda = LDA(n_components=39)
  # fit on data and transform it
  train_data_transformed = lda.fit_transform(train_data, train_labels)
  test_data_transformed = lda.transform(test_data)
  return train_data_transformed, test_data_transformed

# applying lda 
train_data_transformed, test_data_transformed = lda_sklearn()

# KNN
accuracy = knnClassifier(5, train_data_transformed, test_data_transformed)  

print('> accuracy is', accuracy * 100, '%')
