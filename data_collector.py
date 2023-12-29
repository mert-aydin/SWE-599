from github import Github


class DataCollector:
    def __init__(self, token, repo_name):
        """
        Initialize the DataCollector with a GitHub access token and repository name.

        :param token: A GitHub access token for authentication.
        :param repo_name: The full name of the GitHub repository (e.g., 'owner/repo').
        """
        self.github = Github(token)
        self.repo = self.github.get_repo(repo_name)

    def get_commits(self):
        """
        Fetch commit data from the repository.

        :return: A list of commit data.
        """
        commits = self.repo.get_commits()
        return [{'message': commit.commit.message, 'commit_date': commit.commit.author.date, 'url': commit.html_url}
                for commit in commits if not commit.commit.message.startswith('Merge')]

    def get_issues(self, state='all'):
        """
        Fetch issue data from the repository.

        :param state: The state of the issues to fetch ('open', 'closed', or 'all').
        :return: A list of issue data.
        """
        issues = self.repo.get_issues(state=state)
        return [{'id': issue.id, 'title': issue.title, 'body': issue.body, 'created_at': issue.created_at,
                 'closed_at': issue.closed_at, 'url': issue.html_url} for issue in issues if
                issue.pull_request is None]
