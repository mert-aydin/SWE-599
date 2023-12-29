import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_heatmap(data, title, x_label, y_label, file_name):
    """
    Plot a heatmap.

    :param data: 2D data array for heatmap.
    :param title: Title of the plot.
    :param x_label: Label of X-axis.
    :param y_label: Label of Y-axis.
    :param file_name: Name of the file to save the
    """
    sns.heatmap(data, annot=True, cmap='coolwarm')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(file_name)
    plt.show()


def plot_histogram(data, title, x_label, y_label, file_name):
    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Convert the created_at and closed_at columns to datetime
    df['created_at'] = pd.to_datetime(df['created_at'])
    df['closed_at'] = pd.to_datetime(df['closed_at'])

    # Calculate the time taken to close each issue in days
    df['time_to_close'] = (df['closed_at'] - df['created_at']).dt.days

    # Create a histogram of the time taken to close issues
    plt.figure(figsize=(10, 6))
    plt.hist(df['time_to_close'], bins='auto', color='skyblue', alpha=0.7)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    plt.savefig(file_name)
    plt.show()


def create_table(similarity_matrix, commits, issues, similarity_threshold):
    commit_to_issue_matches = []
    for commit_index, scores in enumerate(similarity_matrix):
        highest_similarity_index = scores.argmax()
        highest_similarity_score = scores[highest_similarity_index]

        if highest_similarity_score >= similarity_threshold:
            commit_text = commits['message'][commit_index]
            issue_text = issues['title'][highest_similarity_index]
            issue_link = issues['url'][highest_similarity_index]
            commit_to_issue_matches.append({
                'Commit': commit_text,
                'Matched Issue': issue_text,
                'Issue Link': issue_link,
                'Similarity Score': highest_similarity_score,
            })

    pd.set_option('display.max_colwidth', None)
    matches_df = pd.DataFrame(commit_to_issue_matches)

    # Sort the DataFrame by the 'Similarity Score' column in descending order
    matches_df = matches_df.sort_values(by='Similarity Score', ascending=False)

    # Reset the index if you want to reindex the DataFrame
    matches_df = matches_df.reset_index(drop=True)

    matches_df.to_csv('issue_commit_matches.csv', index=False)
