import pandas as pd
base = "sparse-gradient/phase3_profile_local_v2"
cases = pd.read_csv(f"{base}/profile_cases.csv")
wins = pd.read_csv(f"{base}/case_winners.csv")
print(cases[["case_id","operation","status","correctness","median_ms","profile_sort_basis","kernel_ms_per_invocation","gpu_busy_fraction","likely_host_bound"]])
print()
print(wins)
print()
print("failures:", (cases["status"] != "ok").sum())
print("correctness failures:", (cases["correctness"] == "failed").sum())