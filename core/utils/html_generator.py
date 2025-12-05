"""Утилита для генерации HTML из plotly figure"""
import plotly.graph_objects as go


def fig_to_html(fig: go.Figure) -> str:
    """Конвертирует plotly figure в HTML строку для одного изображения.
    
    Args:
        fig: plotly.graph_objects.Figure
        
    Returns:
        str: HTML строка
    """
    return fig.to_html(
        include_plotlyjs='cdn',
        div_id="plotly-div",
        config={"responsive": True}
    )

