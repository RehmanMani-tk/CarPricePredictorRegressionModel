#%%
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# %%

df = pd.read_csv('carprice.csv')
df.head()
df1= df.copy()


# %%

def exploring_data_shape(dataframe):
    dataframe.columns = dataframe.columns.str.strip()
    dataframe.info()
    print(f'\nthe no. of rows and columns {dataframe.shape}')
    print(f'\nnulls are : {dataframe.isnull().sum()}')
    print(f'\ndulicates are: {dataframe.drop(columns=['car_ID']).duplicated().sum()}')

exploring_data_shape(df1)    

# %%

df1['brand'] = df1['CarName'].str.split(' ').str[0].str.lower()

# %%

df1['brand'].unique()

# %%
sorted(df1['brand'].unique())

# %%

from rapidfuzz import process

list_of_correct_car_brand_names = ['toyota', 'mazda',
'audi','bmw','buick','chevrolet','alfa-romero','dodge','honda','isuzu',
'jaguar','mercury','mitsubishi','nissan','peugeot','plymouth','porsche','renault','saab'
,'subaru','volkswagen','volvo']

for x in df1['brand'].unique():
    match,score,_ = process.extractOne(x,list_of_correct_car_brand_names)
    if score >= 50 and x!=match:
        print(f'{x},{match},{score}')


# %%

brand_corrections = {
    'maxda': 'mazda',
    'porcshce': 'porsche',
    'toyouta': 'toyota',
    'vokswagen': 'volkswagen',
    'vw': 'volkswagen'
}

df1['brand'] = df1['brand'].replace(brand_corrections)
df1['brand'].unique()


# %%
df1.info()

# %%
df1.select_dtypes(include='number').hist(figsize=(20, 20), bins=20)
plt.tight_layout()
plt.show()

# %%

sns.histplot(df1['price'],kde=True)
plt.show()

# %%

corr = df1.select_dtypes(include='number').corr()
sns.heatmap(corr, cmap='coolwarm', center=0)
plt.show()


#%%

cols = ['wheelbase','carlength','carwidth','carheight',
        'curbweight','enginesize','price']
df[cols].corr().round(2)

# %%

plt.figure(figsize=(14,6))
sns.boxplot(data=df1, x='brand', y='price')
plt.xticks(rotation=90)
plt.show()


# %%

categorical_cols = ['brand', 'fueltype', 'aspiration', 'doornumber', 'carbody', 
                     'drivewheel', 'enginelocation', 'enginetype',
                       'cylindernumber', 'fuelsystem']

df_encoded = pd.get_dummies(df1, columns=categorical_cols, drop_first=True)
# %%

df_encoded.shape

#%%

df_encoded.columns.tolist()

# %%

df_encoded = df_encoded.drop(columns=['car_ID', 'CarName', 'wheelbase',
                                       'carlength', 'carwidth', 'curbweight'])
df_encoded.shape

# %%
from sklearn.model_selection import train_test_split

X = df_encoded.drop(columns=['price'])
y = df_encoded['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                     random_state=42)

# %%

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


# %%

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5

print(f"R²: {r2:.3f}")
print(f"RMSE: {rmse:.2f}")

# %%
from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')

print(cv_scores)
print(f"Mean R²: {cv_scores.mean():.3f}")
print(f"Std R²: {cv_scores.std():.3f}")

#%%

sns.boxplot(x=df1['price'])
plt.show()

# %%

Q1 = df1['price'].quantile(0.25)
Q3 = df1['price'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df1[(df1['price'] < lower_bound) | (df1['price'] > upper_bound)]
print(outliers[['brand', 'enginesize', 'horsepower', 'price']].to_string())
print(f"\nNumber of outliers: {len(outliers)}")
# %%

# log-transform the target only — features (X) stay untouched
y_log = np.log(y)

# re-split using the log target
X_train, X_test, y_train_log, y_test_log = train_test_split(X, y_log, test_size=0.3, random_state=42)

model_log = LinearRegression()
model_log.fit(X_train, y_train_log)

# predictions come out in log-dollars — convert back to real dollars before evaluating
y_pred_log = model_log.predict(X_test)
y_pred_real = np.exp(y_pred_log)
y_test_real = np.exp(y_test_log)   # convert the true test values back too, for a fair comparison

r2 = r2_score(y_test_real, y_pred_real)
mse = mean_squared_error(y_test_real, y_pred_real)
rmse = mse ** 0.5

print(f"R² (log model, evaluated in real $): {r2:.3f}")
print(f"RMSE (log model, evaluated in real $): {rmse:.2f}")

# cross-validation on the log-transformed target
cv_scores_log = cross_val_score(model_log, X, y_log, cv=5, scoring='r2')
print(cv_scores_log)
print(f"Mean R² (log): {cv_scores_log.mean():.3f}")
print(f"Std R² (log): {cv_scores_log.std():.3f}")

# %%
coefficients = pd.Series(model_log.coef_, index=X.columns).sort_values(ascending=False)
print(coefficients.head(10))
print(coefficients.tail(10))
# %%
