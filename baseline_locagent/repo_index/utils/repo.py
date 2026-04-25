import logging
import os
import subprocess
import time

logger = logging.getLogger(__name__)


def setup_github_repo(repo: str, base_commit: str, base_dir: str = '/tmp/repos') -> str:
    repo_name = get_repo_dir_name(repo)
    repo_url = f'https://github.com/{repo}.git'

    path = f'{base_dir}/{repo_name}'
    logger.info(
        f"Prepare repo {repo_url} at {path} and checkout commit {base_commit}"
    )
    if not os.path.exists(path):
        os.makedirs(path)
        logger.info(f"Directory '{path}' was created.")
    maybe_clone(repo_url, path)
    checkout_commit(path, base_commit)

    return path


def get_repo_dir_name(repo: str):
    return repo.replace('/', '_')


def _dir_size_bytes(path: str) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for file_name in files:
            try:
                total += os.path.getsize(os.path.join(root, file_name))
            except OSError:
                continue
    return total


def _format_size(num_bytes: int) -> str:
    units = ['B', 'KB', 'MB', 'GB', 'TB']
    size = float(num_bytes)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            if unit == 'B':
                return f"{int(size)}{unit}"
            return f"{size:.1f}{unit}"
        size /= 1024.0
    return f"{int(num_bytes)}B"


def _run_clone_with_progress(clone_cmd, repo_dir, progress_label: str):
    timeout_seconds = int(os.environ.get("LOCAGENT_GIT_CLONE_TIMEOUT", "600"))
    process = subprocess.Popen(
        clone_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    start_time = time.time()
    while True:
        elapsed = int(time.time() - start_time)
        if elapsed >= timeout_seconds:
            process.kill()
            stdout, stderr = process.communicate()
            raise TimeoutError(
                f"{progress_label} exceeded clone timeout of {timeout_seconds}s. "
                f"Current size was {_format_size(_dir_size_bytes(repo_dir))}. "
                f"stderr: {stderr.strip()}"
            )
        try:
            return_code = process.wait(timeout=5)
            break
        except subprocess.TimeoutExpired:
            current_size = _format_size(_dir_size_bytes(repo_dir))
            logger.info(
                f"{progress_label} still running after {elapsed}s/{timeout_seconds}s; current size is {current_size}"
            )

    stdout, stderr = process.communicate()
    if return_code != 0:
        raise subprocess.CalledProcessError(
            return_code, clone_cmd, output=stdout, stderr=stderr
        )
    return stdout, stderr


def maybe_clone(repo_url, repo_dir):
    if os.path.exists(f'{repo_dir}/.git'):
        logger.info(f"Reusing existing repo at '{repo_dir}'")
        return

    logger.info(
        f"Cloning repo from remote source '{repo_url}' into '{repo_dir}'"
    )
    clone_cmd = ['git', 'clone', repo_url, repo_dir]

    stdout, stderr = _run_clone_with_progress(
        clone_cmd,
        repo_dir,
        f"Cloning working repo into '{repo_dir}'",
    )

    if os.path.exists(os.path.join(repo_dir, '.git')):
        logger.info(f"Repo '{repo_url}' was cloned to '{repo_dir}'")
    else:
        logger.info(f"Failed to clone repo '{repo_url}' to '{repo_dir}'")
        raise ValueError(f"Failed to clone repo '{repo_url}' to '{repo_dir}'")


def pull_latest(repo_dir):
    subprocess.run(
        ['git', 'pull'],
        cwd=repo_dir,
        check=True,
        text=True,
        capture_output=True,
    )


def clean_and_reset_state(repo_dir):
    subprocess.run(
        ['git', 'clean', '-fd'],
        cwd=repo_dir,
        check=True,
        text=True,
        capture_output=True,
    )
    subprocess.run(
        ['git', 'reset', '--hard'],
        cwd=repo_dir,
        check=True,
        text=True,
        capture_output=True,
    )


def create_branch(repo_dir, branch_name):
    try:
        subprocess.run(
            ['git', 'branch', branch_name],
            cwd=repo_dir,
            check=True,
            text=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        print(e.stderr)
        raise e


def create_and_checkout_branch(repo_dir, branch_name):
    try:
        branches = subprocess.run(
            ['git', 'branch'],
            cwd=repo_dir,
            check=True,
            text=True,
            capture_output=True,
        ).stdout.split('\n')
        branches = [branch.strip() for branch in branches]
        if branch_name in branches:
            subprocess.run(
                ['git', 'checkout', branch_name],
                cwd=repo_dir,
                check=True,
                text=True,
                capture_output=True,
            )
        else:
            subprocess.run(
                ['git', 'checkout', '-b', branch_name],
                cwd=repo_dir,
                check=True,
                text=True,
                capture_output=True,
            )  # output =
    except subprocess.CalledProcessError as e:
        print(e.stderr)
        raise e


def commit_changes(repo_dir, commit_message):
    subprocess.run(
        ['git', 'commit', '-m', commit_message, '--no-verify'],
        cwd=repo_dir,
        check=True,
        text=True,
        capture_output=True,
    )


def checkout_branch(repo_dir, branch_name):
    subprocess.run(
        ['git', 'checkout', branch_name],
        cwd=repo_dir,
        check=True,
        text=True,
        capture_output=True,
    )


def push_branch(repo_dir, branch_name):
    subprocess.run(
        ['git', 'push', 'origin', branch_name, '--no-verify'],
        cwd=repo_dir,
        check=True,
        text=True,
        capture_output=True,
    )


def get_diff(repo_dir):
    output = subprocess.run(
        ['git', 'diff'], cwd=repo_dir, check=True, text=True, capture_output=True
    )

    return output.stdout


def stage_all_files(repo_dir):
    subprocess.run(
        ['git', 'add', '.'], cwd=repo_dir, check=True, text=True, capture_output=True
    )


def checkout_commit(repo_dir, commit_hash):
    try:
        subprocess.run(
            ['git', 'reset', '--hard', commit_hash],
            cwd=repo_dir,
            check=True,
            text=True,
            capture_output=True,
        )  # output =
    except subprocess.CalledProcessError as e:
        logger.error(e.stderr)
        raise e


def setup_repo(repo_url, repo_dir, branch_name='master'):
    maybe_clone(repo_url, repo_dir)
    clean_and_reset_state(repo_dir)
    checkout_branch(repo_dir, branch_name)
    pull_latest(repo_dir)


def clean_and_reset_repo(repo_dir, branch_name='master'):
    clean_and_reset_state(repo_dir)
    checkout_branch(repo_dir, branch_name)
    pull_latest(repo_dir)
