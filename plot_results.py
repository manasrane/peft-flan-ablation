import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("experiments/results.csv")

# For plotting, use accuracy for classification, rouge1 for summarization
df['performance'] = df['accuracy'].fillna(df['rouge1'])

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='trainable_params', y='performance', hue='method', style='task')
plt.xscale('log')
plt.xlabel('Trainable Parameters (log scale)')
plt.ylabel('Performance')
plt.title('Parameter Efficiency vs Performance')
plt.legend()
plt.savefig('experiments/performance_plot.png')
plt.show()