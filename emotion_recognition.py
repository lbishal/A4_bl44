import cv2
import os
from skimage.feature import hog
from sklearn.svm import SVC
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split, ShuffleSplit, StratifiedKFold
from sklearn.metrics import  accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd

# Use four pre-trained classifiers for face detection
face_detector_1 = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')
face_detector_2 = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_alt.xml')
face_detector_3 = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_alt2.xml')
face_detector_4 = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_alt_tree.xml')


emotion_labels = {'Neutral': 0,
                  'Anger': 1,
                  'Surprise': 2,
                  'Sadness': 3,
                  'Happy': 4}

# add your photos to the folder and set your netid
NetID = 'bl44'


def feature_extraction(img, orientations=16, pixels_per_cell=(16, 16), cells_per_block=(1, 1)):
  """ The function does the following tasks to extract emotion-related features:
      (1) Face detection (2) Cropping the face in the image (3) Resizing the image and (4) Extracting HOG vector.

    Args:
      img: The raw image.
      orientations: The number of bins for different orientations.
      pixels_per_cell: The size of each cell.
      cells_per_block: The size of the block for block normalization.

    Returns:
      features: A HOG vector is returned if face is detected. Otherwise 'None' value is returned.
  """
  

  # If the image is a color image, convert it into gray-scale image
  if img.shape[2] == 3:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


  face_detection_1 = face_detector_1.detectMultiScale(
    img, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
  face_detection_2 = face_detector_2.detectMultiScale(
    img, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
  face_detection_3 = face_detector_3.detectMultiScale(
    img, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
  face_detection_4 = face_detector_4.detectMultiScale(
    img, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)


  # Go over the results of face detection. Stop at the first detected face,
  face_features = None
  if len(face_detection_1) == 1:
    face_features = face_detection_1
  elif len(face_detection_2) == 1:
    face_features = face_detection_2
  elif len(face_detection_3) == 1:
    face_features = face_detection_3
  elif len(face_detection_4) == 1:
    face_features = face_detection_4
  else:
    print("No face detected!")
    # cv2.imshow('No face detected', img)
    # cv2.waitKey(0)


  if face_features is not None:
      global count
      for x, y, w, h in face_features:
          # Get the coordinates and the size of the rectangle containing face
          img = img[y:y+h, x:x+w]
        
          # Resize all the face images so that all the images have the same size
          img = cv2.resize(img, (350, 350))
          # Uncomment the following two lines to visualize the cropped face image
          # cv2.imshow("Cropped Face", img)
          # cv2.waitKey(0)
        
          # Extract HOG descriptor
          features, hog_img = hog(img, orientations=orientations, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block, visualize=True)
          # Uncomment the following tow lines to visualize HOG
          #cv2.imshow('hog', hog_img)
          #cv2.waitKey(0)
          count += 1
          print("Loading: {:d}%".format(int(count / 50 * 100)))
          return features.reshape(1, -1)

  else:
      return None




if __name__ == "__main__":


  "***Feature Extraction***"
  
  def return_dataset(bin_size,cell_size):
    '''

      Parameters
      ----------
      bin_size : int
          num of orientation
      cell_size : int
          num pixels per size

      Returns
      -------
      dataset dictionary keyed by subject name

    '''
    # Dictionary whose <key, value> is <user, (features, labels)>
    dataset = dict()    
    path = './images'
   
    # Get all the folder of individuad subject
    for subject in os.listdir(path):
      if subject[0] == '.':
        continue
      print(subject)
      count = 0
      emotion_dirs = os.listdir(path + '/%s' %subject)
      feature_matrix = None
      labels = None
    
      for emotion_dir in emotion_dirs:
        if emotion_dir[0] == '.':
          continue
        # Get the index associated with the emotion
        emotion_label = emotion_labels[emotion_dir]

        for f in os.listdir(path + '/%s/%s' %(subject, emotion_dir)):
          img = cv2.imread(path + '/%s/%s/' %(subject, emotion_dir) + f)
          # Uncomment the following two lines to visualize the raw images
          # cv2.imshow("raw img", img)
          # cv2.waitKey(0)

          # Extract HOG features        
          features = feature_extraction(img, orientations=bin_size, pixels_per_cell=(cell_size, cell_size), cells_per_block=(1, 1))

          if features is not None:
            feature_matrix = features if feature_matrix is None else np.append(feature_matrix, features, axis=0)
            labels = np.array([emotion_label]) if labels is None else np.append(labels, np.array([emotion_label]), axis=0)
    
      dataset[subject] = (feature_matrix, labels)
    return dataset
 
  "***Person-dependent Model***"
  dataset = return_dataset(10, 16)
  X, y = dataset[NetID]

  # TODO: Use the HOG descriptors to classify different emotions (facial expressions).
  # Here, X is the matrix of HOG descriptors with number of rows equal to the number of images.
  # y is a vector of emotion labels whose length is equal to the number of images.
  skfold= StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
  #svc = SVC(kernel='rbf') #Important params: C is default set to 1.0, gamma is scaled automatically 
  rf = RandomForestClassifier(n_estimators=101,random_state=42)
  all_pred_labels = []
  all_test_labels = []
  for train_idx, test_idx in skfold.split(X,y):
      train_feat, train_labels = X[train_idx,:], y[train_idx]
      test_feat, test_labels   = X[test_idx,:], y[test_idx]
      #fit the classifier
      #svc.fit(train_feat,train_labels)
      rf.fit(train_feat,train_labels)
      pred_labels = rf.predict(test_feat)
      #add the predicted and true labels for analysis
      all_pred_labels.append(pred_labels)
      all_test_labels.append(test_labels)
  all_pred_labels_np = np.concatenate(all_pred_labels)
  all_test_labels_np = np.concatenate(all_test_labels)
  #compute overall and per-class accurracy/precision/recall
  #overall metrics
  accu_overall = accuracy_score(all_test_labels_np,all_pred_labels_np)   
  prec_overall = precision_score(all_test_labels_np, all_pred_labels_np,average='weighted')
  rec_overall  = recall_score(all_test_labels_np, all_pred_labels_np,average='weighted')
  #per class
  prec_perclass = precision_score(all_test_labels_np,all_pred_labels_np,average=None)
  rec_perclass  = recall_score(all_test_labels_np,all_pred_labels_np,average=None)

  "***Person-independent Models***"
  # TODO: Use the model trained on your data to predict another person's emotion.
  #retrain the model on the full training data from my images
  
  rf.fit(X,y)
  #obtain a test subject
  test_subject = 'P1'
  X_test, y_test = dataset[test_subject]
  #X_debug, y_debug = dataset['P1']
  #svc.fit(X_debug, y_debug)
  pred_test = rf.predict(X_test)
  
  #compute confusion matrix, accuracy, and precision/recall per class
  accu_overall_indep = accuracy_score(y,pred_test)   
  print('Print: {0}'.format(accu_overall_indep))
  prec_overall_indep = precision_score(y,pred_test,average='weighted')
  rec_overall_indep  = recall_score(y,pred_test,average='weighted')
  confu_mat_indep    = confusion_matrix(y,pred_test)
  #per class
  prec_perclass_indep = precision_score(y,pred_test,average=None)
  rec_perclass_indep  = recall_score(y,pred_test,average=None)
  #save the confusion matrix
  confu_mat_indep_pd = pd.DataFrame(confu_mat_indep)
  confu_mat_indep_pd.index = ['Neutral','Anger','Surprise','Sadness','Happy']
  confu_mat_indep_pd.columns = ['Neutral','Anger','Surprise','Sadness','Happy']
  confu_mat_indep_pd.to_excel('confu_mat_indep.xlsx')
  

  # TODO: Use leave-one-subject-out cross validation to evaluate the generalized (person-independent) models.
  # You will need to train a model on data from different sets of people and predict the remaining person's emotion.
  def return_accuracy_loso(dataset):
      all_subjects = ['P1','P2','P3','P4','P5','bl44']
      all_pred_loso = []
      all_test_loso = []
      for test_subject in all_subjects:
          X_test, Y_test = dataset[test_subject]
          X_train_, Y_train_ = [], []
          for each_train in list(set(all_subjects) - set([test_subject])):
              X_curr, Y_curr = dataset[each_train]
              X_train_.append(np.copy(X_curr))
              Y_train_.append(np.copy(Y_curr))
          X_train, Y_train = np.concatenate(X_train_), np.concatenate(Y_train_)
          #re-train the model
          rf.fit(X_train,Y_train)
          pred_test = rf.predict(X_test)
          all_pred_loso.append(pred_test)
          all_test_loso.append(Y_test)
      all_pred_loso_np = np.concatenate(all_pred_loso)
      all_test_loso_np = np.concatenate(all_test_loso)
      #compute confusion matrix, accuracy, and precision/recall per class
      accu_overall_loso  = accuracy_score(all_test_loso_np,all_pred_loso_np)   
      prec_overall_loso  = precision_score(all_test_loso_np,all_pred_loso_np,average='weighted')
      rec_overall_loso   = recall_score(all_test_loso_np,all_pred_loso_np,average='weighted')
      confu_mat_loso     = confusion_matrix(all_test_loso_np,all_pred_loso_np)
      #per class
      prec_perclass_loso = precision_score(all_test_loso_np,all_pred_loso_np,average=None)
      rec_perclass_loso  = recall_score(all_test_loso_np,all_pred_loso_np,average=None)
      return accu_overall_loso, prec_overall_loso, rec_overall_loso, prec_perclass_loso, rec_perclass_loso
  
  #compute for the default dataset
  accu_overall_loso, prec_overall_loso, rec_overall_loso, prec_perclass_loso, rec_perclass_loso = return_accuracy_loso(dataset)
  
  
  #effect of bin-size and cell-size on performance
  bin_sizes  = [8,16,32,64]
  cell_sizes = [4,8,16,32,64]
  
  all_accu_results = []
  for bin_size in bin_sizes:
      for cell_size in cell_sizes:
          #retrieve dataset
          curr_dataset = return_dataset(bin_size,cell_size)
          #compute LOSO accuracy
          accu_loso, _,_,_,_ = return_accuracy_loso(curr_dataset.copy())
          all_accu_results.append({'bin_size':bin_size,'cell_size':cell_size,
                                   'accuracy':accu_loso}.copy())
  all_accu_results_pd = pd.DataFrame(all_accu_results)
  
  
  #Person Identification, use neutral class to identify a person using random 5-fold cv
  subj_map = {'P1':0,'P2':1,'P3':2,'P4':3,'P5':4,'bl44':5}
  X_pi, Y_pi = [], []
  for each_subj in subj_map.keys():
      X_curr, Y_curr = dataset[each_subj]
      #only retain neutral class (class 0)
      X_retain = X_curr[np.where(Y_curr==0)[0],:]
      #labels are subjects
      Y_retain = subj_map[each_subj] * np.ones(X_retain.shape[0])
      X_pi.append(np.copy(X_retain))
      Y_pi.append(np.copy(Y_retain))
  X_pi_np = np.concatenate(X_pi)
  Y_pi_np = np.concatenate(Y_pi)
  assert(X_pi_np.shape[0] == Y_pi_np.shape[0])
  assert(X_pi_np.shape[0] == 60) #10 image per neutral class, 6 subjects
  
  all_pred_pi = []
  all_test_pi = []
  for train_idx, test_idx in skfold.split(X_pi_np,Y_pi_np):
      X_train_pi, Y_train_pi = X_pi_np[train_idx,:], Y_pi_np[train_idx]
      X_test_pi, Y_test_pi   = X_pi_np[test_idx,:], Y_pi_np[test_idx]
      rf.fit(X_train_pi,Y_train_pi)
      pred_pi = rf.predict(X_test_pi)
      all_pred_pi.append(pred_pi)
      all_test_pi.append(Y_test_pi)
  all_pred_pi_np = np.concatenate(all_pred_pi)
  all_test_pi_np = np.concatenate(all_test_pi)
  
  #compute confusion matrix, accuracy, and precision/recall per class
  accu_overall_pi  = accuracy_score(all_test_pi_np,all_pred_pi_np)   
  prec_overall_pi  = precision_score(all_test_pi_np,all_pred_pi_np,average='weighted')
  rec_overall_pi   = recall_score(all_test_pi_np,all_pred_pi_np,average='weighted')
  confu_mat_pi     = confusion_matrix(all_test_pi_np,all_pred_pi_np)
  #per class
  prec_perclass_pi = precision_score(all_test_pi_np,all_pred_pi_np,average=None)
  rec_perclass_pi  = recall_score(all_test_pi_np,all_pred_pi_np,average=None)
 
      
 
  
      
   
