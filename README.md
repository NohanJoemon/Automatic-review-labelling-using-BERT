# Automatic Review Labelling using BERT

![image](https://user-images.githubusercontent.com/62956111/173849762-845b3a71-e680-4925-b505-bf6d15bccbd8.png)
<br><br>
&nbsp;&nbsp; Reviews are essential means of knowing the performance of a product. 
In this project, I have created a model that predicts the score of a review based on the text. 
This sentiment analysis model classifies the text into 1 to 5, based on the sentiment behind the review. 
For example, "Nice product" usually means a score of 5 and “Poor quality” usually means a score of 1.
<br><br>
&nbsp;&nbsp; The model was trained using the 
<a href="https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews">Amazon food reviews dataset</a>, 
which contains around 5 lakh reviews. Since there was a class imbalance, I did undersampling to balance the classes. 
I used the BERT model and a linear layer at the end. Therefore, for word embedding, I used the BERT tokenizer.
 The parameters of the BERT model were frozen during the training process to avoid computational complexity. 
 The test accuracy turned out to be 47.4%, much greater than the random case (20%).
 
 
 ## Website preview
 
https://review-labelling.herokuapp.com/

(Note: The app is just below the limit of heroku, hence it can sometimes give error. In those cases, try refreshing it one or two times)
 
 
![image](https://user-images.githubusercontent.com/62956111/173854093-3e940fc3-b4da-4943-8ce0-1dff68610b49.png)
