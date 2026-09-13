# Working MVP: Adaptive Recovery Assistant

## What this version actually estimates

The program does not infer a hidden physiological state or output a fatigue
percentage. It estimates **risk of accumulated work strain** from explainable
signals:

1. **Work context** — active input, passive screen work, absence, break,
   returning after a break.
2. **Continuous session duration** — time spent in active or passive work
   without a meaningful break.
3. **Personal typing changes** — only after the personal five-day baseline is
   ready. The engine requires sustained changes in several checks and does not
   treat one unusual minute as fatigue.
4. **Self-report** — fatigue 0–10 and sleepiness 1–9 entered by the user.
5. **Eye metrics** — displayed for diagnostics only. They do not affect the
   recommendation until blink detection is manually labelled and validated.

## Launch

Fast UI test:

```powershell
python assistant_app.py --demo
```

Real timing:

```powershell
python assistant_app.py
```

## Demo timing

The demo mode intentionally compresses time:

- eye/screen rest after about 45 seconds;
- microbreak after about 70 seconds;
- longer recovery break after about 105 seconds;
- a 15-second manual break resets the demo session.

These are interface-test values only. The real mode uses 45, 70 and 105
minutes, and a meaningful break of 3 minutes.

## Privacy

The MVP does not save camera frames or typed text. It stores local aggregate
metrics, recommendations, self-reports and feedback in SQLite.
