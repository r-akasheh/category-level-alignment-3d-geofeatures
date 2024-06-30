import os
import re


def rename_files(folder_path):
    # Regular expression to match filenames ending with "00X" before the file extension
    pattern = re.compile(r'(.*)00(\d+)(\.\w+)$')

    for filename in os.listdir(folder_path):
        match = pattern.match(filename)
        if match:
            # Construct the new filename
            new_filename = f"{match.group(1)}10{match.group(2)}{match.group(3)}"

            # Full paths for renaming
            old_file_path = os.path.join(folder_path, filename)
            new_file_path = os.path.join(folder_path, new_filename)

            # Rename the file
            os.rename(old_file_path, new_file_path)
            print(f"Renamed: {filename} to {new_filename}")


# Example usage
# folder_path = ('C:/master/robot-vision-modul/FCGF_spconv/dataset/housecat_6d/FCGF_data/housecat_6d_projection/'
#                'npz_files_data_augmentation4/shoe')
# rename_files(folder_path)


def duplicate_and_replace(file_path, output_file_path):
    # Regular expression to match filenames ending with "00X" before the file extension
    pattern = re.compile(r'(\S*?00\d+)(\.\w+)')

    with open(file_path, 'r') as file:
        lines = file.readlines()

    modified_lines = []

    for line in lines:
        modified_lines.append(line.strip())  # Keep original line

        # Split the line into components and replace "00X" with "10X"
        components = line.strip().split()
        new_components = []
        for component in components:
            new_component = re.sub(r'00(\d+)', r'10\1', component)
            new_components.append(new_component)

        modified_lines.append(" ".join(new_components))  # Add modified line

    # Write the result to the output file
    with open(output_file_path, 'w') as file:
        for modified_line in modified_lines:
            file.write(modified_line + '\n')


# Example usage

# for file in os.listdir('C:/master/robot-vision-modul/FCGF_spconv/dataset/housecat_6d/FCGF_data/housecat_6d_plane_teapot_test/'):
#     input_file_path = 'C:/master/robot-vision-modul/FCGF_spconv/dataset/housecat_6d/FCGF_data/housecat_6d_plane_teapot_test/' + file  # Replace with the path to your input file
#     output_file_path = 'C:/master/robot-vision-modul/FCGF_spconv/dataset/housecat_6d/FCGF_data/housecat_6d_plane_teapot_test/' + file  # Replace with the path to your output file
#     duplicate_and_replace(input_file_path, output_file_path)


def modify_lines_in_files(folder_path):
    # Regular expression to match "00X" where X is a single digit
    pattern = re.compile(r'00(\d)')

    # Iterate through all files in the given folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # Process only text files
        if os.path.isfile(file_path) and filename.endswith('.txt'):
            with open(file_path, 'r') as file:
                lines = file.readlines()

            # Create a list to hold the new lines to append
            new_lines = []

            for line in lines:
                modified_line = re.sub(pattern, r'10\1', line.strip())
                new_lines.append(modified_line + '\n')  # Prepare the modified line with newline character

            # Append the modified lines to the file
            with open(file_path, 'a') as file:
                for new_line in new_lines:
                    file.write(new_line)
            print(f"Modified lines appended to {filename}")

#modify_lines_in_files("C:/master/robot-vision-modul/FCGF_spconv/dataset/housecat_6d/FCGF_data/housecat_6d_projection_test/")

def delete_lines_in_files(folder_path):
    # Regular expression to match "10X" where X is greater than 4 (i.e., 105, 106, 107, 108, 109)
    pattern = re.compile(r'10[5-9]')

    # Iterate through all files in the given folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # Process only text files
        if os.path.isfile(file_path) and filename.endswith('.txt'):
            with open(file_path, 'r') as file:
                lines = file.readlines()

            # Create a list to hold the lines that don't match the pattern
            new_lines = []

            for line in lines:
                if not pattern.search(line.strip()):
                    new_lines.append(line)  # Add line if it doesn't contain the pattern

            # Write the remaining lines back to the file
            with open(file_path, 'w') as file:
                file.writelines(new_lines)
            print(f"Modified {filename}")

delete_lines_in_files('C:/master/robot-vision-modul/FCGF_spconv/dataset/housecat_6d/FCGF_data/housecat_6d_projection_test/')