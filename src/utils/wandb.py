from typing import Optional

import wandb


def init(
    team_name: Optional[str],
    project_name: Optional[str],
    run_name: Optional[str],
    key: Optional[str],
) -> None:
    if key is None:
        wandb.login()
    else:
        wandb.login(key=key)

    if project_name is None:
        project_name = input("Please input project name:")

    if team_name is None:
        team_name = input("Please input team name:")

    wandb.init(project=project_name, name=run_name, entity=team_name)
