import argparse
import os

def dump_files(paths, extensions):
    for path in paths:
        for root, _, files in os.walk(path):
            for file in files:
                if extensions is None or len(extensions) == 0 or any(file.endswith(ext) for ext in extensions):
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r') as infile:
                        print(f'<file: {file_path}>')
                        print(infile.read())
                        print('</file>')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dump files with specified extensions from given paths.')
    parser.add_argument('paths', type=str, nargs='+', help='Paths to search for files.')
    parser.add_argument('--extensions', type=str, nargs='+', help='List of file extensions to include.')

    args = parser.parse_args()

    dump_files(args.paths, args.extensions)
