import pandas as pd
import altair as alt
from altair import Chart
from typing import List


def plot_importance(
    features: List[str],
    importance: List[float],
    target: str='',
    num: int=10
) -> Chart:
    source = pd.DataFrame(
        data = {'feature': features, 'importance': importance}
    ).iloc[:num]

    base = alt.Chart(
        data = source, 
        title = f'Target = {target}'
    )

    fig = base.mark_bar(
        color = '#EA4A54'
    ).encode(
        x = alt.X(
            'importance:Q', 
            title = 'Importance (%)'
        ),
        y = alt.Y(
            'feature:N',
            title = None, 
            sort='-x'
        ),
        tooltip = [
            alt.Tooltip(
                'feature', 
                title = 'Feature'
            ),
            alt.Tooltip(
                'importance', 
                title = 'Importance'
            )
        ]
    )

    fig = fig.properties(
        width = 480,
        height = 480
    ).configure_title(
        fontSize = 20,
    ).configure_axis(
        labelFontSize = 15,
        titleFontSize = 20
    )
    return fig