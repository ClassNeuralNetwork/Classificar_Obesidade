import pandas as pd
from sklearn.model_selection import train_test_split
from classificacao_obesidade import df

x = df[['Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight', 'FAVC', 'NCP', 'CAEC', 'SMOKE']]
y = df['NObeyesdad']

input_train, input_test, output_train, output_test = train_test_split(x, y, test_size=0.2)

input_train = pd.DataFrame(input_train)
input_train.to_csv('./train/input_train.csv', index=False)

input_test = pd.DataFrame(input_test)
input_test.to_csv('./test/input_test.csv', index=False)

output_train = pd.DataFrame(output_train)
output_train.to_csv('./train/output_train.csv', index=False)

output_test = pd.DataFrame(output_test)
output_test.to_csv('./test/output_test.csv', index=False)