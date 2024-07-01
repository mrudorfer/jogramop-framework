"""
script to fix paths in all the yaml/urdf files.
converts backslashes into forward slash that works uniformly on all systems.

usage: python fix_paths.py
"""
import os, fnmatch


def replace_backslashes(directory, file_pattern, find='\\', replace='/'):
    for path, dirs, files in os.walk(os.path.abspath(directory)):
        for filename in fnmatch.filter(files, file_pattern):
            filepath = os.path.join(path, filename)
            with open(filepath) as f:
                s = f.read()
            fixed_s = s.replace(find, replace)
            if fixed_s == s:
                print(f'no change in {filepath}')
                continue

            with open(filepath, 'w') as f:
                f.write(fixed_s)
            print(f'fixed {filepath}')


if __name__ == '__main__':
    replace_backslashes('./scenarios', '*.yaml')
    replace_backslashes('./object_library', '*.yaml')
    replace_backslashes('./object_library', '*.urdf')
    replace_backslashes('./robots', '*.urdf')
