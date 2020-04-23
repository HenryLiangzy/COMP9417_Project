# Files in folder:
---
1. count_vector.joblib

	`CountVectorizer` fitted to text_train, no tuning

2. df_train.joblib

	`pd.DataFrame` containing training data with `label` column inserted into raw dataset

3. export_list.joblib

	containing a dictionary mapping filenames in this list to corresponding python variable names

4. label_encoder.joblib

	Label encoder fitted to `df_train.topic`
	
5. text_train.joblib
	
	`pd.Series` containing `df_train.article_words`

6. tf-idf.joblib
	
	`TfidfVectorizer` fitted to text_train, no tuning

7. y_train.joblib
    
	`pd.Series` containing `df_train.label`

8. Readme.md

	This file
