# MyWardrobe
Manage your own wardrobe: You can automise , rather efficiently organise your wardrobe.

# Contents:
Dataset:
  - Train-test dataset in images/final consists of 9 classes:
      -Chaniya choli
      -Collegewear tops
      -Dresses
      -Dungarees
      -Jackets
      -Onepieces
      -Pants
      -Sarees
      -Skirts
  - Wardrobe(testing dataset) in images/test all final consists of random 61 images of clothes, which can be identified using the code         given in final_project_code.py
  
  The images classification is done using KNN Random Forest Classifier. 
  Various features are extracted such as histogram, harlick moments,energy,entropy,etc
  
  # Python libraries used:
    -os
    -csv
    -glob
    -numpy
    -pandas
    -matplotlib
    -opencv
    -scikit learn 
    -mahotas
    
  # Limitations:
    The accuracy of current model is not high enough, i.e at times, the model is unable to correctly able to identify the class an           images belongs to.
    
  # Improvements:
     Other techniques such as 
          -CNN(which needs GPU, due to which i was unable to implement this) 
          -Transfer learning(which will be used in the near future)
          -One shot learning
  
