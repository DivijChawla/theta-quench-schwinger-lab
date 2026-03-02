# Study Registry

Each study file in this directory must define:

- `study_id`
- `primary_endpoint`
- `power_target`
- `models`
- `size_grid`
- `regime_grid`

Optional `phases` contain executable run specs for `stage1`, `stage2`, and/or `stage3`.

Run a study phase with:

```bash
python3 -m tqm.run --study <study_id> --phase stage1|stage2|stage3
```

Dry-run command generation:

```bash
python3 -m tqm.run --study <study_id> --phase stage3 --dry-run
```
