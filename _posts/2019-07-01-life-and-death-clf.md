---
layout: post
title: Life and Death Classification
---

One of the more "insightful" projects that I did during my Metis program is a scaled-down version of Kaggle's pet adotpion competition. [link here](https://www.kaggle.com/c/petfinder-adoption-prediction)

While not personally owner of any furry friends, I've always had yearning to break down the cute facade of house pets (in this case, dogs) and see what lies beneath! Alas, cuteness comes in many flavors and is more in the eye of the beholder so the project turns out to be a good education of what people value in dogs at least as they make their decision to adopt (or not).

The rather unsavory fact was never too far off in the background, however, that if shelter animals stay in the shelter beyond a certain number of days without adoption, they are most likely destined for a tragic end--there are millions of stray pets being euthanized just in the United States. A life-and-death situation indeed.

## A Much Simplified Data Processing Step

To keep the project feasible within the time constraint, I chose to ignore the photos collection keyed to the individual pet ID's. A few pets with missing sentiment analysis of profiles were dropped. And instead of a multi-class classficication of how soon an adoption occurs (1 month, 2 months, etc.), a binary target of whether adopted within 100 days (the longest timeframe in the data) or not was created.

```python
train_df = pd.read_csv('data/train.csv')
columns_to_drop = ['Description', 'VideoAmt', 'RescuerID', 'Name', 'Quantity']
train_df = train_df[train_df['Quantity'] == 1].drop(columns=columns_to_drop)
train_sent = pd.read_csv('sent_train.csv', index_col=0)

# keep only the records that have sentiment score
train_df = train_df.merge(train_sent, how='inner', left_on='PetID', right_index=True)
train_df['AdoptionSpeed'] = train_df['AdoptionSpeed'].replace({1: 0, 2: 0, 3: 0, 4: 1})

# one hot encode the dummy variables stored as values
onehot_columns = ['Type', 'Breed1', 'Breed2', 'Gender', 'Color1', 'Color2', 'Color3',
                  'Vaccinated', 'Dewormed', 'Sterilized', 'Health', 'State']
train_df2 = pd.get_dummies(train_df, columns=onehot_columns)

# test train split
X_train, X_test, label_train, label_test = train_test_split(train_df2.drop(columns=['AdoptionSpeed', 'PetID']),
                                                            train_df2['AdoptionSpeed'],
                                                            test_size=0.3, random_state=41)

# standard scaler
std = StandardScaler()
std.fit(X_train)
X_train = std.transform(X_train)
X_test = std.transform(X_test)
```

Part in the post to tell the page how to title your post and how to render it.

Below are some examples of loading images, making links, and doing other
markdown-y things.


[This is a link](http://thisismetis.com)

![Image test]({{ site.url }}/images/AlanLeeShireGandalf.JPG)

### Other things
* Like
* lists
* and 
* stuff
