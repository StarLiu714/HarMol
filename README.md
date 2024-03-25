# HarMol: AI assisted Recipe Property Prediction Model

## Preliminary Test

### MolSets 1077 Dataset

1. Arrhenius Equation re-construct:
   ```python
   def fit_and_predict_conductivity(group_data, temp_to_predict):
     # Transformations
     group_data['1/T'] = 1 / group_data['temperature']
     group_data['log_sigma'] = group_data['conductivity']
     # Linear regression features and target
     X = group_data[['1/T']]
     y = group_data['log_sigma']
     # Fit
     model = LinearRegression()
     model.fit(X, y)
     # Predict
     inverse_temp = 1 / temp_to_predict
     predicted_log_sigma = model.predict([[inverse_temp]])
     predicted_log_sigma_col = predicted_log_sigma[0]
     return predicted_log_sigma_col
 ```
