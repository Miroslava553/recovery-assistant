from app_core.state_machine import PresenceSignals, UserStateManager


def show(time_sec: float, manager: UserStateManager, signals: PresenceSignals) -> None:
    decision = manager.update(signals, now=time_sec)
    print(
        f"{time_sec:>5.1f} с | {decision.state.value:<12} | "
        f"усталость анализируется: {decision.fatigue_analysis_allowed} | "
        f"{decision.reason}"
    )


manager = UserStateManager(
    transition_debounce_sec=1.0,
    away_face_threshold_sec=5.0,
    away_input_threshold_sec=5.0,
    baseline_warmup_sec=4.0,
)
manager.reset(now=0.0)

# Пользователь печатает.
show(
    0.0,
    manager,
    PresenceSignals(True, 0.5, True, 0.95),
)
show(
    1.1,
    manager,
    PresenceSignals(True, 0.5, True, 0.95),
)

# Пользователь читает: мышь не двигается, но лицо и взгляд на экране.
show(
    3.0,
    manager,
    PresenceSignals(True, 12.0, True, 0.95),
)
show(
    4.1,
    manager,
    PresenceSignals(True, 12.0, True, 0.95),
)

# Пользователь отошёл: лица нет и ввода нет.
show(
    5.0,
    manager,
    PresenceSignals(False, 20.0, None, 0.80),
)
show(
    10.2,
    manager,
    PresenceSignals(False, 25.0, None, 0.80),
)
show(
    11.3,
    manager,
    PresenceSignals(False, 26.0, None, 0.80),
)

# Пользователь вернулся: работа засчитывается сразу, идёт только
# разогрев личной нормы печати.
show(
    12.0,
    manager,
    PresenceSignals(True, 0.2, True, 0.95),
)
show(
    14.0,
    manager,
    PresenceSignals(True, 0.2, True, 0.95),
)
show(
    16.2,
    manager,
    PresenceSignals(True, 0.2, True, 0.95),
)
show(
    17.3,
    manager,
    PresenceSignals(True, 0.2, True, 0.95),
)
