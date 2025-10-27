import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_emotion_distribution_change(*, period_dfs, column="emotion_group", title=None):
    """
    Visualize the relative frequency of grouped emotions across different time periods.
    """
    # Count relative frequencies
    frames = []
    for period, df in period_dfs.items():
        counts = df[column].value_counts(normalize=True).reset_index()
        counts.columns = ['grouped_emotion', 'relative_frequency']
        counts['Period'] = period
        frames.append(counts)

    combined_df = pd.concat(frames, ignore_index=True)

    # Reorder x-axis by overall frequency
    ordered_emotions = (
        combined_df
        .groupby('grouped_emotion')['relative_frequency']
        .sum()
        .sort_values(ascending=False)
        .index
        .tolist()
    )
    combined_df['grouped_emotion'] = pd.Categorical(
        combined_df['grouped_emotion'], 
        categories=ordered_emotions, 
        ordered=True
    )

    # Define color palette
    palette = {
        'Before Riot': '#a6a6a6',
        'Riot Window': '#d62728',
        'After Riot': '#1f77b4',
    }

    plt.figure(figsize=(14, 6))
    ax = sns.barplot(
        data=combined_df,
        x='grouped_emotion',
        y='relative_frequency',
        hue='Period',
        palette=palette
    )

    # --- Add percentage labels on top of bars ---
    for container in ax.containers:
        ax.bar_label(
            container,
            fmt='%.2f',  # format as percentage
            label_type='edge',
            padding=3,
            fontsize=9,
            color='black'
        )

    plt.title(title or 'Relative Frequency of Emotions Around the Capitol Riot', fontsize=14)
    plt.xlabel('Emotion', fontsize=12)
    plt.ylabel('Relative Frequency', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Period')
    plt.tight_layout()
    plt.show()

    return combined_df


def plot_emotions_by_party_and_period(
    *,
    period_dfs: dict,
    column: str = "emotion_group",
    title: str | None = None,
    emotion_order: list[str] | None = None,
    show_percent_labels: bool = False,
    savepath: str | None = None,
):
    """
    Plot the distribution of emotions by political party and time period.

    Parameters
    ----------
    period_dfs : dict
        A dictionary mapping period names to DataFrames.
    column : str
        The column name containing emotion labels.
    title : str or None
        An optional title for the plot.
    emotion_order : list[str] or None
        Custom order for emotions. If None, the order is determined by frequency.
    show_percent_labels : bool
        Whether to display percentage labels on the bars.
    savepath : str or None
        Path to save the plot as an image. If None, the plot is not saved.
    """
    frames = []
    for period_name, df in period_dfs.items():
        d = df.copy()
        d["Period"] = period_name
        frames.append(d)
    combined = pd.concat(frames, ignore_index=True)
    combined = combined[combined["party"].isin(["D", "R"])]

    # Count
    grouped = (
        combined.groupby(["party", "Period", column], observed=True)
        .size()
        .reset_index(name="count")
        .rename(columns={column: "Emotion", "party": "Party"})
    )
    grouped["Party"] = grouped["Party"].map({"D": "Democrats", "R": "Republicans"})

    # Consistent emotion order (user-provided or data-driven)
    if emotion_order is None:
        emotion_order = (
            grouped.groupby("Emotion")["count"].sum().sort_values(ascending=False).index.tolist()
        )

    all_periods = list(period_dfs.keys())
    period_order = [p for p in ["Before Riot", "Riot Window", "After Riot"] if p in all_periods] or all_periods

    full_index = pd.MultiIndex.from_product(
        [list(map(str, grouped["Party"].unique())), period_order, emotion_order],
        names=["Party", "Period", "Emotion"],
    )
    grouped = (
        grouped.set_index(["Party", "Period", "Emotion"])
        .reindex(full_index, fill_value=0)
        .reset_index()
    )

    # Relative frequencies within Party×Period
    grouped["relative_frequency"] = (
        grouped.groupby(["Party", "Period"], observed=True)["count"]
        .transform(lambda x: x / max(x.sum(), 1))
    )

    grouped["Emotion"] = pd.Categorical(grouped["Emotion"], categories=emotion_order, ordered=True)
    grouped["Period"] = pd.Categorical(grouped["Period"], categories=period_order, ordered=True)

    palette_default = {"Before Riot": "#a6a6a6", "Riot Window": "#d62728", "After Riot": "#1f77b4"}
    palette = {p: palette_default.get(p, "#8c8c8c") for p in period_order}

    g = sns.catplot(
        data=grouped,
        x="Emotion",
        y="relative_frequency",
        hue="Period",
        col="Party",
        kind="bar",
        palette=palette,
        height=5.5,
        aspect=1.25,
        sharey=True,
        legend_out=True,
    )

    # --- axes formatting + labels on bars ---
    for ax in g.axes.flatten():
        ax.set_xlabel("Emotion")
        ax.set_ylabel("Relative Frequency" + (" (%)" if show_percent_labels else ""))
        ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", rotation=40)

        # Add numbers on top of bars (works for grouped bars per hue)
        if show_percent_labels:
            # fix y-limit for percent display
            ax.set_ylim(0, 1.05)
            for container in ax.containers:
                labels = [f"{rect.get_height()*100:.0f}%" if rect.get_height() > 0 else "" 
                          for rect in container]
                ax.bar_label(container, labels=labels, label_type='edge', padding=2, fontsize=8)
        else:
            # raw decimals if wanted
            ymax = grouped["relative_frequency"].max()
            ax.set_ylim(0, ymax * 1.15 if ymax > 0 else 1.0)
            for container in ax.containers:
                labels = [f"{rect.get_height():.2f}" if rect.get_height() > 0 else "" 
                          for rect in container]
                ax.bar_label(container, labels=labels, label_type='edge', padding=2, fontsize=8)

    g.set_titles("{col_name}")
    g.figure.subplots_adjust(top=0.87, bottom=0.18, right=0.85)
    g.figure.suptitle(title or "Emotion Frequencies by Party and Period", fontsize=14, y=0.97)

    if savepath:
        g.figure.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.show()

    return grouped



def plot_emotion_shift_by_party(
    *,
    period_dfs,
    column="emotion_group",
    reference_period="Before Riot",
    top_n=None,
    show_percent_labels: bool = False,
):
    """
    Analyze and visualize shifts in emotion frequencies by political party across time periods.

    Parameters
    ----------
    period_dfs : dict
        A dictionary mapping period names to DataFrames.
    column : str
        The column name containing emotion labels.
    reference_period : str
        The baseline period for calculating shifts.
    top_n : int or None
        If specified, limits the plot to the top-N emotions with the largest shifts.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the calculated emotion shifts.
    """
    frames = []
    for period_name, df in period_dfs.items():
        d = df.copy()
        d["Period"] = period_name
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df = df[df["party"].isin(["D", "R"])].rename(columns={"party": "Party", column: "Emotion"})
    df["Party"] = df["Party"].map({"D": "Democrats", "R": "Republicans"})

    grouped = (
        df.groupby(["Party", "Period", "Emotion"], observed=True)
          .size().reset_index(name="count")
    )
    grouped["relative_frequency"] = (
        grouped.groupby(["Party", "Period"], observed=True)["count"]
               .transform(lambda x: x / max(x.sum(), 1))
    )

    pivot = grouped.pivot_table(
        index=["Party", "Emotion"], columns="Period", values="relative_frequency", fill_value=0
    )
    if reference_period not in pivot.columns:
        raise ValueError(f"Reference period '{reference_period}' not found in input.")

    delta_frames = []
    for period in pivot.columns:
        if period == reference_period:
            continue
        delta = pivot[[reference_period, period]].copy()
        delta["Delta"] = delta[period] - delta[reference_period]
        delta = delta.reset_index()[["Party", "Emotion", "Delta"]]
        delta["DeltaPeriod"] = period
        delta_frames.append(delta)

    emotion_shift = pd.concat(delta_frames, ignore_index=True)

    if top_n:
        kept = []
        for (party, comp), g in emotion_shift.groupby(["Party", "DeltaPeriod"]):
            kept.append(g.loc[g["Delta"].abs().nlargest(top_n).index])
        emotion_shift = pd.concat(kept, ignore_index=True)

    present = emotion_shift["DeltaPeriod"].unique().tolist()
    preferred = ["Riot Window", "After Riot"]
    col_order = [p for p in preferred if p in present] or present
    emotion_shift["DeltaPeriod"] = pd.Categorical(emotion_shift["DeltaPeriod"],
                                                  categories=col_order, ordered=True)

    g = sns.catplot(
        data=emotion_shift,
        x="Emotion", y="Delta",
        hue="Party", col="DeltaPeriod", col_order=col_order,
        kind="bar",
        palette={"Democrats": "#4285F4", "Republicans": "#EA4335"},
        height=6, aspect=1.2, sharey=True, legend_out=True,
    )
    g.set_xticklabels(rotation=45)
    g.set_axis_labels("Emotion", "Δ Relative Frequency")
    g.set_titles("Shift During {col_name}")

    for ax in g.axes.flat:
        ax.axhline(0, color="black", linestyle="--", linewidth=1)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        ax.set_axisbelow(True)

        # ✅ labels on top of the bars (handles grouped bars per hue)
        for container in ax.containers:
            if show_percent_labels:
                labels = [f"{h*100:+.0f}%" if (h := rect.get_height()) != 0 else "" for rect in container]
            else:
                labels = [f"{h:+.2f}" if (h := rect.get_height()) != 0 else "" for rect in container]
            ax.bar_label(container, labels=labels, label_type='edge', padding=2, fontsize=8)

    g.fig.subplots_adjust(top=0.86, bottom=0.2, right=0.87)
    g.fig.suptitle(f"Emotion Frequency Shifts by Party (relative to {reference_period})", y=0.98)

    plt.show()
    return emotion_shift




