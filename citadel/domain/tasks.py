"""Select exactly one configured atomic task."""
def profile_for(resources: dict, profiles: dict):
    actions = {s["action_id"] for s in resources["steps"]}
    matches = [(name, profile) for name, profile in profiles.items()
               if actions and actions <= set(profile["action_ids"])
               and (not profile.get("task_codes") or resources.get("task_code") in profile["task_codes"])]
    if len(matches) != 1:
        raise ValueError("Task actions need exactly one configured atomic-task profile")
    name, profile = matches[0]
    if not all(profile.get(k) for k in ("success", "allowed", "failures")):
        raise ValueError("Task profile needs success, allowed variation and failure rules")
    hold, tolerance = profile.get("hold_seconds"), profile.get("hold_tolerance_s", 0)
    if hold is not None and not (0 <= tolerance < hold):
        raise ValueError("Invalid hold duration/tolerance")
    return {"name": name, **profile}
