import pandas as pd

# 读取CSV文件
df = pd.read_csv('adam_test_results.csv')

# 计算"Deviation"列的分位数
percentiles = [0.1, 0.2, 0.25, 0.75, 0.8, 0.9]
quantiles = df['Deviation'].quantile(percentiles)

print(quantiles)

# 计算在-3到+3之间的值的占比
in_range_count = df['Deviation'][(df['Deviation'] >= -5) & (df['Deviation'] <= 5)].count()
total_count = df['Deviation'].count()
proportion = in_range_count / total_count

print("\n-3到+3范围内的占比：")
print(proportion)
