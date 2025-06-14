import os
import pstats

def combine_and_print_stats(directory):
    """
    Combine all .prof files in the given directory and print the combined stats.
    
    Parameters:
        directory (str): Path to the directory containing .prof files.
    """
    # Initialize a Stats object
    combined_stats = None

    # Iterate over all .prof files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".prof"):
            filepath = os.path.join(directory, filename)
            print(f"Loading stats from: {filepath}")
            
            # Load the stats file
            if combined_stats is None:
                combined_stats = pstats.Stats(filepath)
            else:
                combined_stats.add(filepath)
    
    if combined_stats is None:
        print("No .prof files found in the directory.")
        return

    # Clean up paths for readability and sort the stats
    combined_stats.strip_dirs()
    combined_stats.sort_stats("cumulative")

    # Print the combined stats
    print("\nCombined Profile Stats:\n")
    combined_stats.print_stats()

# Usage example
if __name__ == "__main__":
    # Set the directory containing .prof files
    stats_directory = "."  # Change this to your directory path

    # Combine and print stats
    combine_and_print_stats(stats_directory)
