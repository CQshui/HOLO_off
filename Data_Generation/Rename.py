import os


def batch_rename_files(directory_path, new_prefix):
    os.chdir(directory_path)
    files = os.listdir()
    for old_name in files:
        new_name = new_prefix + old_name
        os.rename(old_name, new_name)


if __name__ == '__main__':
    path = r'F:\Data\20240711\mengtuoshi\single\mengtuoshi'
    prefix = 'a_'
    batch_rename_files(path, prefix)
