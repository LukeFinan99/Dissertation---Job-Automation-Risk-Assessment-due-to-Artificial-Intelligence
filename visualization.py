import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Sample Data
np.random.seed(42)
data = np.random.normal(loc=1950, scale=50, size=5000)
boxplot_data = [np.random.normal(loc=val, scale=15, size=100) for val in range(1900, 2000, 20)]

# Set a larger figure size
fig, axes = plt.subplots(3, 2, figsize=(20, 18))

# Top 15 Features Influencing Automation Risk
axes[0, 0].barh(['Feature ' + str(i) for i in range(15)], np.random.rand(15), color='c')
axes[0, 0].set_title("Top 15 Features Influencing Automation Risk", fontsize=14)
axes[0, 0].set_xlabel("Feature Importance", fontsize=12)
axes[0, 0].set_ylabel("Features", fontsize=12)
axes[0, 0].tick_params(axis='both', labelsize=10)
axes[0, 0].grid(True, linestyle='--', alpha=0.6)

# Distribution of Job Stability Scores
sns.histplot(data, kde=True, ax=axes[0, 1], bins=30, color='blue')
axes[0, 1].set_title("Distribution of Job Stability Scores", fontsize=14)
axes[0, 1].set_xlabel("Job Stability Score", fontsize=12)
axes[0, 1].set_ylabel("Frequency", fontsize=12)
axes[0, 1].tick_params(axis='both', labelsize=10)
axes[0, 1].grid(True, linestyle='--', alpha=0.6)

# PCA Visualization
scatter_x = np.random.normal(0, 10, 100)
scatter_y = np.random.normal(0, 10, 100)
axes[1, 0].scatter(scatter_x, scatter_y, c=np.random.rand(100), cmap='viridis', edgecolors='black', alpha=0.75)
axes[1, 0].set_title("PCA Visualization of Job Characteristics", fontsize=14)
axes[1, 0].set_xlabel("First Principal Component", fontsize=12)
axes[1, 0].set_ylabel("Second Principal Component", fontsize=12)
axes[1, 0].tick_params(axis='both', labelsize=10)
axes[1, 0].grid(True, linestyle='--', alpha=0.6)

# Correlation Heatmap
corr_matrix = np.random.rand(10, 10)
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', ax=axes[1, 1], fmt=".2f", linewidths=0.5)
axes[1, 1].set_title("Correlation Heatmap of Top Features", fontsize=14)

# Boxplot of Automation Risk Across Industries
sns.boxplot(data=boxplot_data, ax=axes[2, 0], palette='Set3')
axes[2, 0].set_title("Automation Risk Distribution Across Industries", fontsize=14)
axes[2, 0].set_xlabel("Industry Index", fontsize=12)
axes[2, 0].set_ylabel("Automation Risk Score", fontsize=12)
axes[2, 0].tick_params(axis='both', labelsize=10)
axes[2, 0].grid(True, linestyle='--', alpha=0.6)

# Adjust spacing to reduce overlap
plt.subplots_adjust(wspace=0.35, hspace=0.5, top=0.92, left=0.08, right=0.96, bottom=0.08)
plt.suptitle("Job Automation Risk Analysis Visualizations", fontsize=18, fontweight='bold')

# Save figure to file
plt.savefig(r"job_automation_comprehensive_visualization.png", dpi=300, bbox_inches='tight')
plt.show()
