import time
import timeit
from tqdm import tqdm
from rich.progress import (
    track,
    Task,
    Progress,
    ProgressColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
    FileSizeColumn,
    TotalFileSizeColumn,
    DownloadColumn,
    TransferSpeedColumn,
    SpinnerColumn,
    RenderableColumn,
    TaskProgressColumn,
)
from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
from rich.box import DOUBLE


# Setup code for no-bar scenario
no_bar_setup = """N = 100_000"""
no_bar_code = """
for i in range(N):
    # Simulate work
    x = i * i
"""

# Setup code for tqdm scenario
tqdm_setup = """
from tqdm import tqdm
N = 100_000
"""
tqdm_code = """
for i in tqdm(range(N), desc='tqdm loop', leave=False):
    x = i * i
"""

# Setup code for Rich scenario
rich_setup = """
from rich.progress import Progress
N = 100_000
"""
rich_code = """
with Progress(transient=True) as progress:
    task_id = progress.add_task("rich loop", total=N)
    for i in range(N):
        x = i * i
        progress.update(task_id, advance=1)
"""


def compare_progress_bars():
    no_bar_time = timeit.timeit(no_bar_code, setup=no_bar_setup, number=1000)
    tqdm_time = timeit.timeit(tqdm_code, setup=tqdm_setup, number=1000)
    rich_time = timeit.timeit(rich_code, setup=rich_setup, number=1000)

    print(f"No progress bar: {no_bar_time:.5f} seconds")
    print(f"tqdm progress bar: {tqdm_time:.5f} seconds")
    print(f"rich progress bar: {rich_time:.5f} seconds")


class EmojiProgressColumn(ProgressColumn):
    def render(self, task: Task) -> Text:
        # No total? Just show a thinking emoji
        if task.total is None:
            return Text("🤔", style="dim")

        progress_ratio = task.completed / task.total if task.total else 0

        if progress_ratio < 0.3:
            emoji = "🐌"
        elif progress_ratio < 0.7:
            emoji = "🏃"
        elif progress_ratio < 1.0:
            emoji = "🚀"
        else:
            emoji = "🎉"

        return Text(emoji, style="bold magenta")


def main() -> None:
    # tqdm progress bar
    # for _ in tqdm(range(1000)):
    #     time.sleep(0.01)
    # import sys; sys.exit(0)

    # Basic rich progress bar
    # for _ in track(range(1000)):
    #     time.sleep(0.01)

    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        # *Progress.get_default_columns() # Equivalent to the previous 4 columns
    )
    task1 = progress.add_task("Working...", total=1000)
    task2 = progress.add_task("Working hard...", total=1000)
    # print(f"{task1=}")
    # print(f"{task2=}")

    progress_custom = Progress(
        TextColumn("[progress.description]{task.description}"),
        FileSizeColumn(),  # assumes step size is bytes
        TotalFileSizeColumn(),
        SpinnerColumn(spinner_name="christmas", finished_text="🎁"),
        BarColumn(),
        MofNCompleteColumn(),
        EmojiProgressColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        expand=True,
    )
    task3 = progress_custom.add_task("Playing hard...", total=1000)
    grouped_progress = Group(progress, Panel(progress_custom, box=DOUBLE))
    with Live(grouped_progress):
        for _ in range(1000):
            progress.update(task1, advance=1)
            progress.update(task2, advance=2)
            progress_custom.update(task3, advance=1)
            time.sleep(0.01)


if __name__ == "__main__":
    # compare_progress_bars()
    main()
