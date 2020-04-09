# Files in folder:
---
1. count_vector.joblib

	`CountVectorizer` fitted to text_train, no tuning

2. df_train.joblib

	`pd.DataFrame` containing training data with `label` column inserted into raw dataset

3. label_encoder.joblib

	Label encoder fitted to `df_train.topic`
	
4. text_train.joblib
	
	`pd.Series` containing `df_train.article_words`

5. tf-idf.joblib
	
	`TfidfVectorizer` fitted to text_train, no tuning

6. y_train.joblib
    
	`pd.Series` containing `df_train.label`

7. Readme.md

	This file
