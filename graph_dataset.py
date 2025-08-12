import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

train_df = pd.read_csv('./configs/coco_annotations/train/image_tags_ground_truth.csv')
valid_df = pd.read_csv('./configs/coco_annotations/valid/image_tags_ground_truth.csv')
test_df = pd.read_csv('./configs/coco_annotations/test/image_tags_ground_truth.csv')

train_tag_counts = train_df.iloc[:, 1:].sum().sort_values(ascending=False)
plt.figure(figsize=(12, 6))
train_tag_counts.plot(kind='bar', color='skyblue')
plt.title('Tag Distribution in Train Set')
plt.xlabel('Tags')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

train_tag_counts = train_df.iloc[:, 1:].sum().sort_values(ascending=False)
for tag, count in train_tag_counts.items():
    print(f"Train Tag: {tag}, Count: {count}")

valid_tag_counts = valid_df.iloc[:, 1:].sum().sort_values(ascending=False)
plt.figure(figsize=(12, 6))
valid_tag_counts.plot(kind='bar', color='lightgreen')
plt.title('Tag Distribution in Validation Set')
plt.xlabel('Tags')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

valid_tag_counts = valid_df.iloc[:, 1:].sum().sort_values(ascending=False)
for tag, count in valid_tag_counts.items():
    print(f"Validation Tag: {tag}, Count: {count}")

train_tag_counts = train_df.iloc[:, 1:].sum().sort_values(ascending=False)
valid_tag_counts = valid_df.iloc[:, 1:].sum().sort_values(ascending=False)
tags = train_tag_counts.index

x = np.arange(len(tags)) 
width = 0.35
plt.figure(figsize=(12, 6))

# Plot Train vs Validation
plt.bar(x - width/2, train_tag_counts, width, label='Train', color='skyblue')
plt.bar(x + width/2, valid_tag_counts, width, label='Validation', color='lightgreen')
plt.title('Train vs Validation Tag Distribution')
plt.xlabel('Tags')
plt.ylabel('Count')
plt.xticks(x, tags, rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

test_tag_counts = test_df.iloc[:, 1:].sum().sort_values(ascending=False)
plt.figure(figsize=(12, 6))
test_tag_counts.plot(kind='bar', color='salmon')
plt.title('Tag Distribution in Test Set')
plt.xlabel('Tags')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

test_tag_counts = test_df.iloc[:, 1:].sum().sort_values(ascending=False)
for tag, count in test_tag_counts.items():
    print(f"Test Tag: {tag}, Count: {count}")

train_tag_counts = train_df.iloc[:, 1:].sum().sort_index()
valid_tag_counts = valid_df.iloc[:, 1:].sum().sort_index()
test_tag_counts = test_df.iloc[:, 1:].sum().sort_index()

tags = train_tag_counts.index


x = np.arange(len(tags))  
width = 0.35  

fig, axes = plt.subplots(1, 2, figsize=(20, 6), sharey=True)

# Plot Train vs Validation
axes[0].bar(x - width/2, train_tag_counts, width, label='Train', color='skyblue')
axes[0].bar(x + width/2, valid_tag_counts, width, label='Validation', color='lightgreen')
axes[0].set_title('Train vs Validation Tag Distribution')
axes[0].set_xlabel('Tags')
axes[0].set_ylabel('Count')
axes[0].set_xticks(x)
axes[0].set_xticklabels(tags, rotation=45)
axes[0].legend()

# Plot Test tag distribution
axes[1].bar(tags, test_tag_counts, color='salmon')
axes[1].set_title('Test Tag Distribution')
axes[1].set_xlabel('Tags')
axes[1].tick_params(axis='x', rotation=45)
axes[1].legend(['Test'])

# Adjust layout
plt.tight_layout()
plt.show()

df = pd.concat([train_df, valid_df, test_df], ignore_index=True)
Classifications = df.drop('image_filename', axis=1).sum().sort_values(ascending=False)
print(Classifications)