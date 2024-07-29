import pandas as pd
from imblearn.over_sampling import SMOTE

input_train = pd.read_csv('./dataset/train/input_train.csv')
output_train = pd.read_csv('./dataset/train/output_train.csv')

sm = SMOTE(random_state=42)

input_train_balanced, output_train_balanced = sm.fit_resample(input_train, output_train)

input_train_balanced = pd.DataFrame(input_train_balanced)
input_train_balanced.to_csv('./dataset/train_balanced/input_train_balanced.csv', index=False)

output_train_balanced = pd.DataFrame(output_train_balanced)
output_train_balanced.to_csv('./dataset/train_balanced/output_train_balanced.csv', index=False)