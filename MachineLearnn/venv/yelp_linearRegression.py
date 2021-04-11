import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


def model_these_features(feature_list):
    # define ratings and features, with the features limited to our chosen subset of data
    ratings = df.loc[:, 'stars']
    features = df.loc[:, feature_list]

    # perform train, test, split on the data
    X_train, X_test, y_train, y_test = train_test_split(features, ratings, test_size=0.2, random_state=1)

    # don't worry too much about these lines, just know that they allow the model to work when
    # we model on just one feature instead of multiple features. Trust us on this one :)
    if len(X_train.shape) < 2:
        X_train = np.array(X_train).reshape(-1, 1)
        X_test = np.array(X_test).reshape(-1, 1)

    # create and fit the model to the training data
    model = LinearRegression()
    model.fit(X_train, y_train)

    # print the train and test scores
    print('Train Score:', model.score(X_train, y_train))
    print('Test Score:', model.score(X_test, y_test))

    # print the model features and their corresponding coefficients, from most predictive to least predictive
    print(sorted(list(zip(feature_list, model.coef_)), key=lambda x: abs(x[1]), reverse=True))

    # calculate the predicted Yelp ratings from the test data
    y_predicted = model.predict(X_test)

    # plot the actual Yelp Ratings vs the predicted Yelp ratings for the test data
    plt.scatter(y_test, y_predicted)
    plt.xlabel('Yelp Rating')
    plt.ylabel('Predicted Yelp Rating')
    plt.ylim(1, 5)
    plt.show()

businesses = pd.read_json('/Users/huan-chilin/Documents/yelp_regression_project/yelp_business.json', lines = True)
reviews = pd.read_json('/Users/huan-chilin/Documents/yelp_regression_project/yelp_review.json', lines = True)
users = pd.read_json('/Users/huan-chilin/Documents/yelp_regression_project/yelp_user.json', lines = True)
checkins = pd.read_json('/Users/huan-chilin/Documents/yelp_regression_project/yelp_checkin.json', lines = True)
tips = pd.read_json('/Users/huan-chilin/Documents/yelp_regression_project/yelp_tip.json', lines = True)
photos = pd.read_json('/Users/huan-chilin/Documents/yelp_regression_project/yelp_photo.json', lines = True)

pd.options.display.max_columns = 60
pd.options.display.max_colwidth = 500

## print(users.head())

## print(businesses['business_id'])

## selectedID = businesses[businesses['business_id'] == '5EvUIR4IzCWUOm0PsUZXjA']
## print(selectedID['stars'])

combined_Data = pd.merge(businesses, reviews, how= 'left', on='business_id')
combined_Data = pd.merge(combined_Data, users, how= 'left', on='business_id')
combined_Data = pd.merge(combined_Data, checkins, how= 'left', on='business_id')
combined_Data = pd.merge(combined_Data, tips, how= 'left', on='business_id')
combined_Data = pd.merge(combined_Data, photos, how= 'left', on='business_id')

## drop the features that are not binary or continuous
list_of_features_to_remove = ['address','attributes','business_id','categories','city','hours','is_open','latitude',
                              'longitude','name','neighborhood','postal_code','state','time']

## using pands drop, axis = 1 for dropping column, inplace for replacing date frame
combined_Data.drop(list_of_features_to_remove, axis=1, inplace=True)

## check if any data frame is empty
isEmpyt = combined_Data.isna().any()

combined_Data.fillna(value=0, inplace=True)
isEmpyt = combined_Data.isna().any()
## print(isEmpyt)

corr_data = combined_Data.corr()
# print(corr_data['stars'])

# print(combined_Data['average_review_age'].values.reshape(-1, 1))

# plt.scatter(combined_Data['stars'].values.reshape(-1, 1),
#             combined_Data['average_review_age'].values.reshape(-1, 1), alpha=0.6)
# plt.scatter(combined_Data['stars'].values.reshape(-1, 1),
#             combined_Data['average_review_sentiment'].values.reshape(-1, 1), alpha=0.6)
# plt.scatter(combined_Data['stars'].values.reshape(-1, 1),
#             combined_Data['average_review_age'].values.reshape(-1, 1), alpha=0.6)
#
# plt.show()

features = combined_Data[['average_review_length', 'average_review_age']]
rating = combined_Data['stars']

X_train, X_test, Y_train, Y_test = train_test_split(features, rating, test_size=0.2, train_size=0.8, random_state=1)

model = LinearRegression()
model.fit(X_train, Y_train)

trained_score = model.score(X_train, Y_train)
test_score = model.score(X_test, Y_test)
print(trained_score)
print(test_score)

mod