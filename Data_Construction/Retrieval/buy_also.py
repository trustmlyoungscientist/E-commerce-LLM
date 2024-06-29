import pandas as pd
import json
import random
import matplotlib.pyplot as plt
import gc


# Specify the path to the JSON file
json_file_path = 'meta_Grocery_and_Gourmet_Food.json'

# Initialize an empty list to store the parsed JSON objects
json_list = []

# Open the file and read it line by line
with open(json_file_path, 'r', encoding='utf-8') as file:
    for line in file:
        # Parse each line as a JSON object and add it to the list
        json_list.append(json.loads(line))

# Convert the list of JSON objects to a DataFrame
df = pd.DataFrame(json_list)
del json_list
gc.collect()
df = df.drop_duplicates("asin")
df = df[['asin','title','also_buy']]
# Now df is a DataFrame containing all the data from the file
df


# Append the title of also_buy products to the target product


df_filtered = df[~df['also_buy'].isna()]
df_exploded = df_filtered.explode('also_buy')

# Append also_buy
tmp = df_filtered[["asin","title"]]
tmp.columns = ['also_buy',"title_also"]
df_tmp = df_exploded[["asin","title", "also_buy"]]
df_also_buy = pd.merge(df_tmp, tmp, how="left", on = "also_buy")
df_also_buy = df_also_buy[~df_also_buy['title_also'].isna()]
df_also_buy = df_also_buy.reset_index(drop = True)
df_also_buy
# Group by asin
grouped = df_also_buy.groupby('asin')
# Take the top 5 rows of each group, as most of the example data in the official data is hit@3
df_also_buy = grouped.head(5)

# Convert title and title_also to sets
df_also_buy['title_set'] = df_also_buy['title'].apply(lambda x:set(x.split()))
df_also_buy['title_also_set'] = df_also_buy['title_also'].apply(lambda x:set(x.split()))

# Calculate overlap ratio
def calculate_overlap_ratio(row):
    intersection_size = len(row['title_set'].intersection(row['title_also_set']))
    min_size = min(len(row['title_set']), len(row['title_also_set']))
    return intersection_size / min_size if min_size != 0 else 0

overlap_ratio = df_also_buy.apply(calculate_overlap_ratio, axis=1)
print(f"Before removing highly overlapping: {df_also_buy.shape[0]}")
df_also_buy = df_also_buy[overlap_ratio <= 0.65]
print(f"After removing highly overlapping: {df_also_buy.shape[0]}")



# Upon analysis, some titles and title_also have very high overlap, making them meaningless

# # Count the number of entries in each bucket
# bucket_counts = (df_also_buy['overlap_ratio'] //0.1).value_counts().sort_index()

# # Plot a histogram
# plt.bar(bucket_counts.index.astype(str), bucket_counts.values)
# plt.xlabel('Overlap Ratio')
# plt.ylabel('Frequency')
# plt.title('Overlap Ratio Distribution')
# plt.show()


# Randomly sample 14 to append
tmp = df_also_buy[['asin', 'title', 'title_also']]
# Group by asin and combine the values of the title_also column into a list
grouped = tmp.groupby(['asin','title']).agg({'title_also': list}).reset_index()
grouped.rename(columns={'title_also': 'title_also_list'}, inplace=True)

all_title_set = list(df[~df['title'].isna()]['title'])
all_row_title_neg = [random.sample(all_title_set, 15 - len(single_row)) for single_row in grouped['title_also_list']]
all_row_title_neg = pd.Series(all_row_title_neg)

grouped['title_negative_list'] = all_row_title_neg

# Mark positive and negative samples
grouped['title_also_list'] = grouped['title_also_list'].apply(lambda row: [(i, 1) for i in row])
grouped['title_negative_list'] = grouped['title_negative_list'].apply(lambda row: [(i, 0) for i in row])

# Combine positive and negative samples and shuffle their positions
grouped['combined_list'] = grouped.apply(lambda row: row['title_also_list'] + row['title_negative_list'], axis=1)
grouped['combined_list'] = grouped['combined_list'].apply(lambda x: random.sample(x, len(x)))

# Get the indices of positive samples, which should start from 1
grouped['output_field'] = grouped['combined_list'].apply(lambda row: [idx + 1 for idx, i in enumerate(row) if i[1] == 1])

# Only retain the required columns
grouped = grouped[['title', 'combined_list', 'output_field']]
grouped['combined_list'] = grouped['combined_list'].apply(lambda x: "\n".join([f"{idx + 1}. {i[0]}" for idx, i in enumerate(x)]))
grouped.rename(columns={'combined_list': 'candidate_list'}, inplace=True)
print(grouped['candidate_list'][0])
grouped


import random

prompt_all_kind = [
    "A user on an online shopping website has just purchased a product '{}'. The following numbered list contains 15 products. Please select {} products from the list that the user may also purchase.\nProduct List: \n{}"\
    "You should output {} numbers that correspond to the selected products. There should be a comma separating every two numbers. "\
    "Only respond with the results. Do not say any word or explanations.\nOutput: ",
    
    "You are a helpful shop assistant. A user would like to buy the product '{}'. Please select the products that the user may also buy from the following numbered list.\nProduct List: \n{}"\
    "You should output {} numbers that correspond to the selected products. There should be a comma separating every two numbers. "\
    "Only respond with the results. Do not say any word or explanations.\nOutput: "
]

final_df = grouped.sample(frac=1)
all_prompt_list = []
for row in final_df.values:
    title, candidate_list, positive_idx = row[0], row[1], row[2]
    prompt_kind = random.choice([0, 1])
    prompt = prompt_all_kind[prompt_kind]
    if prompt_kind == 0:
        all_prompt_list.append(prompt.format(title, str(len(positive_idx)), candidate_list, str(len(positive_idx)), axis=1))
    else:
        all_prompt_list.append(prompt.format(title, candidate_list, str(len(positive_idx)), axis=1))

final_df['prompt'] = all_prompt_list
final_df


final_df[['prompt', 'output_field']].to_json("example2.json", orient="records", lines=True)
final_df[['prompt', 'output_field']].to_csv("example2.csv", index=None)
