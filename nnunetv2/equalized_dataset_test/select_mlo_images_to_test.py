import pandas as pd
import numpy as np

mlo_dummy_df = pd.read_csv('nnunetv2/equalized_dataset_test/mlo-all-dummy.csv')
paths_df = pd.read_csv('nnunetv2/equalized_dataset_test/path-vitor.csv')


raw_merged_df = pd.merge(mlo_dummy_df, paths_df,how='left', on='Dummy_ID')

raw_merged_df = raw_merged_df[['Dummy_ID','birads_density', 'Subfolder', 'Files']]

# Create a column to make easier filter the MLO side
raw_merged_df['side'] = raw_merged_df['Subfolder'].map(lambda x: 'L' if 'PROC_L' in x else 'R')

# Only left MLOs that have at least one Rigth MLO 
left_merged_df = raw_merged_df[raw_merged_df['side']=='L']
left_merged_df = left_merged_df.sort_values('Files').drop_duplicates('Dummy_ID', keep='first')

# Only right MLOs that have at least one Left MLO
rigth_merged_df = raw_merged_df[raw_merged_df['side']=='R']
rigth_merged_df = rigth_merged_df.sort_values('Files').drop_duplicates('Dummy_ID', keep='first')

# Merging left and right MLOs
merged_df = pd.merge(left_merged_df, rigth_merged_df[['Dummy_ID', 'Files', 'Subfolder']],
                            how='inner', on='Dummy_ID', suffixes=('', '_R'))
# Renaming columns for clarity
merged_df = merged_df.rename(columns={'Files': 'Files_L', 'Subfolder': 'Subfolder_L'})
# Now we have a DataFrame with Dummy_ID, birads_density, Files_L and Files_R


merged_df_birads_1 = merged_df[merged_df['birads_density']==1]
merged_df_birads_2 = merged_df[merged_df['birads_density']==2]
merged_df_birads_3 = merged_df[merged_df['birads_density']==3]
merged_df_birads_4 = merged_df[merged_df['birads_density']==4]

nb_dummy_ids_birads_1 = merged_df_birads_1["Dummy_ID"].unique().shape[0]
nb_dummy_ids_birads_2 = merged_df_birads_2["Dummy_ID"].unique().shape[0]
nb_dummy_ids_birads_3 = merged_df_birads_3["Dummy_ID"].unique().shape[0]
nb_dummy_ids_birads_4 = merged_df_birads_4["Dummy_ID"].unique().shape[0]

print(f'Number of Dummy_IDs with birads_density 1: {nb_dummy_ids_birads_1}')
print(f'Number of Dummy_IDs with birads_density 2: {nb_dummy_ids_birads_2}')
print(f'Number of Dummy_IDs with birads_density 3: {nb_dummy_ids_birads_3}')
print(f'Number of Dummy_IDs with birads_density 4: {nb_dummy_ids_birads_4}')

assert nb_dummy_ids_birads_1 >= 100, f'{nb_dummy_ids_birads_1} < 100 rows'
assert nb_dummy_ids_birads_2 >= 100, f'{nb_dummy_ids_birads_2} < 100 rows'
assert nb_dummy_ids_birads_3 >= 100, f'{nb_dummy_ids_birads_3} < 100 rows'
assert nb_dummy_ids_birads_4 >= 100, f'{nb_dummy_ids_birads_4} < 100 rows'

# random dummy_id
selected_dummy_ids_birads_1 = np.random.choice(nb_dummy_ids_birads_1, size=100, replace=False)
selected_dummy_ids_birads_2 = np.random.choice(nb_dummy_ids_birads_2, size=100, replace=False)
selected_dummy_ids_birads_3 = np.random.choice(nb_dummy_ids_birads_3, size=100, replace=False)
selected_dummy_ids_birads_4 = np.random.choice(nb_dummy_ids_birads_4, size=100, replace=False)

# select dfs
selected_df_birads_1 = merged_df_birads_1.iloc[selected_dummy_ids_birads_1]
selected_df_birads_2 = merged_df_birads_2.iloc[selected_dummy_ids_birads_2]
selected_df_birads_3 = merged_df_birads_3.iloc[selected_dummy_ids_birads_3]
selected_df_birads_4 = merged_df_birads_4.iloc[selected_dummy_ids_birads_4]

# concat dfs
final_df = pd.concat([selected_df_birads_1, selected_df_birads_2,
                        selected_df_birads_3, selected_df_birads_4])

final_df = final_df[['Dummy_ID', 'birads_density', 'Files_L', 'Subfolder_L', 'Files_R', 'Subfolder_R']]

final_df.to_csv('nnunetv2/equalized_dataset_test/random_selected_MLO.csv', index=False)



