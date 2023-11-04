

# import pandas as pd
#
# # Read the TSV file into a DataFrame
# file_path = '.\\raw_data\\gym\\glue-sst2\\_test.tsv'
# df = pd.read_csv(file_path, sep='\t')
#
# # Remove the 'idx' column from the DataFrame
# if 'idx' in df.columns:
#     df.drop(columns=['idx'], inplace=True)
#
# # Save the modified DataFrame back to the same file
# df.to_csv(file_path, sep='\t', index=False)
#
# print(f"Column 'idx' removed and saved to '{file_path}'")





# import pandas as pd
# import json
#
# # Read JSON file and convert each object to TSV format
# input_file_path = '.\\raw_data\\gym\\glue-sst2\\_train.json'
# output_file_path = '.\\raw_data\\gym\\glue-sst2\\_train.tsv'
#
# with open(input_file_path, 'r', encoding='utf-8') as json_file:
#     # Read JSON objects line by line
#     json_objects = [json.loads(line) for line in json_file]
#
#     # Convert each JSON object to a DataFrame and concatenate them
#     dfs = []
#     for json_object in json_objects:
#         df = pd.json_normalize(json_object)  # Convert JSON object to DataFrame
#         dfs.append(df)
#
#     # Concatenate individual DataFrames into a single DataFrame
#     concatenated_df = pd.concat(dfs, ignore_index=True)
#
#     # Save concatenated DataFrame to TSV format in the same file
#     concatenated_df.to_csv(output_file_path, sep='\t', index=False, encoding='utf-8')
#
# print(f"JSON data converted to TSV and saved to '{output_file_path}'")


import torch
# import os
# import torch
#
# # Create a tensor on CPU
# cpu_tensor = torch.tensor([1, 2, 3])
#
# # Move the tensor to GPU (if available)
# gpu_tensor = cpu_tensor.to('cuda')  # 'cuda' refers to the default GPU device
#
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# # Your code that uses GPU memory
# # Move the GPU tensor back to CPU
# cpu_tensor = gpu_tensor.to('cpu')
#
# # Empty the GPU memory
# torch.cuda.empty_cache()
#
# # Continue with your code



a='.\\raw_data\\gym\\glue-sst2\\entail2test_shots_5training_shots_128_result.json'
b='.\\raw_data\\gym\\glue-sst2\\_test.tsv'



# Read data from JSON file
import json

with open(a, 'r', encoding='utf-8') as json_file:
    json_data = [json.loads(line) for line in json_file]

# Read data from TSV file
import pandas as pd

tsv_data = pd.read_csv(b, delimiter='\t', encoding='utf-8')

# Extract labels from JSON data
json_labels = []
for item in json_data:
    out_field = item.get('out', None)
    if out_field and isinstance(out_field, list) and len(out_field) > 0 and out_field[0].isdigit():
        label = int(out_field[0])
        json_labels.append(label)
    else:
        # Handle invalid or missing 'out' field, you can skip this entry or set a default label as per your requirement
        print(f"Invalid 'out' field in JSON data: {item}")
print(json_labels)

# Extract labels from TSV data
tsv_labels = tsv_data['label'].tolist()
print(tsv_labels)
# Calculate accuracy
correct_predictions = sum(x == y for x, y in zip(json_labels, tsv_labels))
total_predictions = len(json_labels)
accuracy = correct_predictions / total_predictions * 100

print(f'Accuracy: {accuracy:.2f}%')


