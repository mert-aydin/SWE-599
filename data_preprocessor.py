import pandas as pd


def preprocess_commits(commits):
    """
    Preprocess a list of commit dictionaries.

    :param commits: List of dictionaries, each representing a commit.
    :return: A DataFrame with preprocessed commit data.
    """
    if not commits or not isinstance(commits, list):
        raise ValueError("Commits should be a list of dictionaries.")

    # Convert the list of dictionaries to a DataFrame
    commits_df = pd.DataFrame(commits)

    # Normalize data, handle missing values, etc.
    commits_df['commit_date'] = pd.to_datetime(commits_df['commit_date'])
    # commits_df['message'] = commits_df['message'].str.lower().str.replace('[^\w\s]', '', regex=True)

    return commits_df


def preprocess_issues(issues):
    """
    Preprocess a list of issue dictionaries.

    :param issues: List of dictionaries, each representing an issue.
    :return: A DataFrame with preprocessed issue data.
    """
    if not issues or not isinstance(issues, list):
        raise ValueError("Issues should be a list of dictionaries.")

    # Convert the list of dictionaries to a DataFrame
    issues_df = pd.DataFrame(issues)

    # Normalize data, handle missing values, etc.
    issues_df['created_at'] = pd.to_datetime(issues_df['created_at'])
    issues_df['closed_at'] = pd.to_datetime(issues_df['closed_at'])
    # issues_df['title'] = issues_df['title'].str.lower().str.replace('[^\w\s]', '', regex=True)
    # issues_df['body'] = issues_df['body'].str.lower().str.replace('[^\w\s]', '', regex=True)

    issues_df['title_body'] = issues_df['title'] + ' ' + issues_df['body']

    return issues_df
