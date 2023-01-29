import numpy as np
import pandas as pd
#1. 데이터
# 데이터 경로
path = 'C:/study/_data/homework/'


# CSV 파일 가져오기
""" samsung_csv = pd.read_csv(path + 'samsung.csv', index_col=0, encoding = 'euc-kr')
amore_csv = pd.read_csv(path + 'amore.csv', index_col=0, encoding = 'euc-kr') """

# 원하는 값 가져오기
df_ss = pd.read_csv(path + 'samsung.csv', encoding='cp949', index_col=0, usecols=[1, 2, 3, 4, 8])
print(df_ss)

df_af = pd.read_csv(path + 'amore.csv', encoding='cp949', index_col=0, usecols=[1, 2, 3, 4, 8])
print(df_af)

# 결측치 제거
print(df_ss.isnull().sum())
samsung_csv = df_ss.dropna()

print(df_af.isnull().sum())
amore_csv = df_af.dropna()

