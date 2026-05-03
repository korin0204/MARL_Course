# 採点ガイド

## Bomber Arena

```bash
.venv/bin/python scripts/evaluate_bomber_tournament.py --submissions submissions --episodes 8
```

教室スクリーン表示:

```bash
.venv/bin/python scripts/evaluate_bomber_tournament.py --submissions submissions --episodes 1 --live
```

pygame がない環境では `--live-ascii` を使います。

## Coop Kitchen

```bash
.venv/bin/python scripts/evaluate_coop_submissions.py --submissions submissions --episodes 3
```

Zero-shot league:

```bash
.venv/bin/python scripts/evaluate_coop_zero_shot.py --submissions submissions --episodes 2
```

主な指標は `avg_score`, `avg_soups`, `avg_collisions`, `avg_invalid_interacts` です。
